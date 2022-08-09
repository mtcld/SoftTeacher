from scipy.optimize import linear_sum_assignment
import torch 
import numpy as np

def iou_loss(boxes1: torch.Tensor,boxes2: torch.Tensor):

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

    return (1 - intsctk /  (unionk + 1e-7)).item()

def l1_loss(box1,box2):
    return abs(box1-box2).sum().item()

def label_loss(label1,label2):
    def get_info(label):
        carpart = label[:label.find('+')]
        side = label[label.find('+')+1:]
        
        return carpart, side
    
    cp1,side1 = get_info(label1)
    cp2,side2 = get_info(label2)
    
    score = 0
    
    if cp1 == cp2 :
        score += 0.5
        
        if side1 == side2:
            score += 0.5
    
    return -score

def loss(label1,box1,label2,box2):
    # loss boxes
    iou_l = iou_loss(box1,box2)
    l1_l = l1_loss(box1,box2)
    
    # loss labels
    label_l = label_loss(label1,label2)
    
    return iou_l+l1_l+label_l

class Hungarian():
    def __init__(self,w,h):
        self.W = w
        self.H = h

    def normalize(self,box):
        xyxy = np.array([box[0],box[2]]).reshape(-1)
        box = [xyxy[0]/self.W,xyxy[1]/self.H,xyxy[2]/self.W,xyxy[3]/self.H]
        return torch.tensor(box)
    
    def bipartite_matching(self,track_info, detect_info,pred_json):
        # base on tracking of previous frame + detect of current frame -> relabel current detection -> output new roi list and pred_json
        track_labels = []
        track_boxes = []

        for label,box,idx in track_info:
            track_labels.append(label)
            track_boxes.append(self.normalize(box))
        
        detect_labels = []
        detect_boxes = []
        detect_ind = []

        for label, box, idx in detect_info:
            detect_labels.append(label)
            detect_boxes.append(self.normalize(box))
            detect_ind.append(idx)
        
        cost_global = []
        
        for idx,dbox in enumerate(detect_boxes):
            row_cost = []
            for idy,tbox in enumerate(track_boxes):
                row_cost.append(loss(detect_labels[idx],dbox,track_labels[idy],tbox))
            cost_global.append(row_cost)
        
        cost_global = np.array(cost_global)
        cost_global = np.pad(cost_global,[(0,int(cost_global.shape[0]<cost_global.shape[1])*abs(cost_global.shape[0]-cost_global.shape[1])),
                                        (0,int(cost_global.shape[0]>cost_global.shape[1])*abs(cost_global.shape[0]-cost_global.shape[1]))],
                                        'constant',constant_values=(10,))

        row_ind,col_ind = linear_sum_assignment(cost_global)

        for r,c in zip(row_ind,col_ind):
            if r >= len(detect_labels) or c >= len(track_labels):
                continue
            
            if cost_global[r,c] < 0.45:
                detect_info[r][0] = track_labels[c]
                pred_json[0]['carpart']['labels'][detect_ind[r]] = track_labels[c]
                print('relabel : ',detect_labels[r], track_labels[c],cost_global[r][c])

                # print('debug pred_json : ',pred_json[0]['carpart']['labels'][detect_ind[r]])

        return detect_info, pred_json
    
    