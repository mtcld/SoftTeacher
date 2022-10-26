from cProfile import label
import cv2
import numpy as np

from video_process.car.view.view import get_front_back_view
from video_process.utils.mask import check_mask_inside_mask,expend_mask,check_overlap,masks_edge,get_contours
from video_process.utils.util import check_carpart_in_list
import torchvision.ops.boxes as bops
import torch


inside_part={'fbu_front_bumper':['grille','flp_front_license_plate','fli_fog_light','fbe_fog_light_bezel'],'door':['window','handle','mirror'],'tyre':['alloy_wheel'],\
            'qpa_quarter_panel':['fuel_tank_door','window'],'tail_gate':['rwi_rear_windshield',],'fender':['sli_side_turn_light'],'mirror':['sli_side_turn_light']}


# correct_masks_labels=[{'target':'grille','bg':['license_plate','hood']},{'target':'fbu_front_bumper','bg':['license_plate','grille','fog_light_bezel']},{'target':'tail_gate','bg':['rwi_rear_windshield']}]
correct_masks_labels=[{'target':'grille','bg':['flp_front_license_plate','hood']},\
{'target':'fbu_front_bumper','bg':['flp_front_license_plate','grille','fog_light_bezel','fli_fog_light']},\
{'target':'tail_gate','bg':['rwi_rear_windshield']},\
{'target':'hood','bg':['fender']},\
{'target':'fender','bg':['fbu_front_bumper']},\
{'target':'door','bg':['mirror','window','handle']}]
no_side_carpart=['hood','grille','tail_gate']

def keep_no_side_carpart(labels,masks,scores,bboxes):
    for i in range(len(labels)-1,-1,-1):
        if  labels[i] not in no_side_carpart:
            scores.pop(i)
            masks.pop(i)
            bboxes.pop(i)
            labels.pop(i)

def only_keep_roof(labels,masks,scores,bboxes):

    if 'roof' not in labels:
        return 

    sort_labels = [x for _, x in sorted(zip(masks, labels), key=lambda pair: cv2.countNonZero(pair[0]))]
    if 'roof'  in sort_labels[-2:]:
        for i in range(len(labels)-1,-1,-1):
            if 'roof' not in labels[i]:
                labels.pop(i)
                scores.pop(i)
                masks.pop(i)
                bboxes.pop(i)

    if len([label for label in sort_labels[-2:] if 'door' in label]) <2:
        return 
    if 'roof' not in sort_labels[-3:]:
        return
    for i in range(len(labels)-1,-1,-1):
        if 'roof' not in labels[i]:
            labels.pop(i)
            scores.pop(i)
            masks.pop(i)
            bboxes.pop(i)

def correct_labels(pred_json):
    
    labels= pred_json['carpart']['labels']
    masks= pred_json['carpart']['masks']

    if check_carpart_in_list('fender',labels) and  check_carpart_in_list('fwi_windshield',labels) and not check_carpart_in_list('hood',labels):
        
        for i in range(len(labels)-1,-1,-1):
            if 'fender' in labels[i]:
                fender = masks_edge(get_contours([masks[i]]),masks)
                for j in range(len(labels)-1,-1,-1):
                    if 'fwi_windshield' in labels[j]:
                        fwi_windshield =masks_edge(get_contours([masks[j]]),masks) 
                        if check_mask_inside_mask(fender,fwi_windshield,0.8,check_max=True):
                            pred_json['carpart']['labels'][i] ='hood+f'
                            break

def correct_masks(labels,masks,correct_masks_label):
    fr_masks= [mask for i,mask in enumerate(masks) if labels[i] in correct_masks_label['target'] ]
    if len(fr_masks) ==0:
        return labels,masks
    backgrounds = [mask for i,mask in enumerate(masks) if labels[i] in correct_masks_label['bg']]
    # print(correct_masks_label['target'],correct_masks_label['bg'])
    if len(backgrounds) > 0:
        bg_mask =backgrounds[0]
        bg_mask = bg_mask.astype(np.uint8)
        for background in backgrounds[1:]:
            background = background.astype(np.uint8)
            bg_mask=cv2.bitwise_or(background,bg_mask)
        bg_mask=cv2.bitwise_not(bg_mask)
        for i,(label, mask) in enumerate(zip(labels,masks)):
            if label in correct_masks_label['target']:
                masks[i]=cv2.bitwise_and(mask.astype(np.uint8),bg_mask)
        
    return labels,masks

def correct_font_back_labels(labels,masks,view=None):
    if view is None:
        view =get_front_back_view(labels)
    for i in range(len(labels)-1,-1,-1):
        if 'fbu_front_bumper' in labels[i] and 'rbu_rear_bumper' not in labels and view < 0 :
            labels[i]='rbu_rear_bumper'

        if 'rbu_rear_bumper' in labels[i] and 'fbu_front_bumper' not in labels and view > 0 :
            labels[i]='fbu_front_bumper'

        if 'hli_head_light' in labels[i] and 'tli_tail_light' not in labels and view < 0 :
            labels[i]='tli_tail_light'  

        if 'tli_tail_light' in labels[i] and 'hli_head_light' not in labels and view > 0 :
            labels[i]='hli_head_light' 
              
        if 'fender' in labels[i] and 'qpa_quarter_panel' not in labels and view < 0 :
            labels[i]='qpa_quarter_panel'
        
        if 'qpa_quarter_panel' in labels[i] and 'fender' not in labels and view > 0 :
            labels[i]='fender'
    
    if labels.count('door')==1 and labels.count('fender')==1 and 'handle' in labels  and 'mirror' not in labels and len(labels)<8  :
        fender_mask =[expend_mask(mask,thickness=40) for i,mask in enumerate(masks) if 'fender'  in  labels[i]][0]
        handle_mask =[expend_mask(mask,thickness=40) for i,mask in enumerate(masks) if 'handle'  in  labels[i]][0]

        if check_overlap(fender_mask,handle_mask):
            for i, label in enumerate(labels):
                if 'fender' in label:
                    labels[i]='qpa_quarter_panel'

    return labels

def merge_bbox(bboxes_i,bboxes_j):
    x_i,y_i,x_i1,y_i1 =bboxes_i[:4]
    x_j,y_j,x_j1,y_j1 =bboxes_j[:4]
    x= min(x_i,x_j)
    y= min(y_i,y_j)

    x1= max(x_i1,x_j1)
    y1= max(y_i1,y_j1) 
    return np.array([x,y,x1,y1])
     
def merge_carpart(labels,masks,scores,bboxes):
    for i in range(len(labels)-1,0,-1):
        if 'door' not in labels[i] and 'window' not in labels[i] and labels.count(labels[i])>1:
            for j in range(i - 1, -1, -1):
                if labels[j]==labels[i]:
                    over_lap=cv2.bitwise_and(expend_mask(masks[i]),expend_mask(masks[j]))
                    if cv2.countNonZero(over_lap) >0:
                        masks[j] =cv2.bitwise_or(masks[i],masks[j])
                        scores[j] =max(scores[i],scores[j])
                        bboxes[j] =merge_bbox(bboxes[i],bboxes[j])
                        labels.pop(i)
                        scores.pop(i)
                        masks.pop(i)
                        bboxes.pop(i)
                        break 
    # for i in range(len(labels)-1,0,-1):
    #     if  labels[i] in ['fbu_front_bumper','rocker_panel'] and labels.count(labels[i])>1:
    #         for j in range(i - 1, -1, -1):
    #             if labels[j]==labels[i]:
    #                 masks[j] =cv2.bitwise_or(masks[i],masks[j])
    #                 scores[j] =max(scores[i],scores[j])
    #                 bboxes[j] =merge_bbox(bboxes[i],bboxes[j])
    #                 labels.pop(i)
    #                 scores.pop(i)
    #                 masks.pop(i)
    #                 bboxes.pop(i)
    #                 break

    # for i in range(len(labels)-1,0,-1):
    #     if  labels[i] in 'fender' and labels.count(labels[i])>1 and labels.count('grille') ==0:
    #         for j in range(i - 1, -1, -1):
    #             if labels[j]==labels[i]:
    #                 masks[j] =cv2.bitwise_or(masks[i],masks[j])
    #                 scores[j] =max(scores[i],scores[j])
    #                 bboxes[j] =merge_bbox(bboxes[i],bboxes[j])
    #                 labels.pop(i)
    #                 scores.pop(i)
    #                 masks.pop(i)
    #                 bboxes.pop(i)
    #                 break        

def filter_overlap_carpart(labels,masks,scores,bboxes):
    # print('debug filter car part: *********************************')
    for i in range(len(labels)-1,0,-1):
        for j in range(i - 1, -1, -1):
            if (labels[i] in inside_part and  labels[j]  in inside_part[labels[i]])\
                 or (labels[j] in inside_part and labels[i] in inside_part[labels[j]]):
                continue
            # print('debug inside mask : ',masks[i].shape,masks[j].shape)
            # masks[i] = masks[i].astype(np.uint8)
            # masks[j] = masks[j].astype(np.uint8)

            if check_mask_inside_mask(masks[i].astype(np.uint8),masks[j].astype(np.uint8),0.7,check_max=False):
                
                # print('kkkkkkkkkkkkk',labels[i],labels[j])
                if 'rocker_panel' in labels[i] or 'rocker_panel' in labels[j]:
                    continue 
                i_area= cv2.countNonZero(masks[i].astype(np.uint8))
                j_area= cv2.countNonZero(masks[j].astype(np.uint8))
                ratio_area = min(i_area,j_area)/max(i_area,j_area)
                if (ratio_area< 0.35 and i_area > j_area) or (ratio_area >= 0.35 and scores[i] >scores[j]) :
                    masks[j] =masks[i]
                    scores[j] =scores[i]
                    labels[j] =labels[i]
                    bboxes[j] =bboxes[i]

                labels.pop(i)
                scores.pop(i)
                masks.pop(i)
                bboxes.pop(i)
                
                break
    labels,masks,scores,bboxes=filter_tyre(labels,masks,scores,bboxes)
    # print('labelslabelslabelslabels',labels)
    return labels,masks,scores,bboxes


neighbor_relation={'tyre':['qpa_quarter_panel','fender','fbu_front_bumper','rbu_rear_bumper','door'] ,'handle':['door']}
def filter_tyre(labels,masks,scores,bboxes):
    for key in neighbor_relation.keys():
        for i in range(len(labels)-1,-1,-1):
            if key not in labels[i]:
                continue 
            
            neibor_list= []
            for j in range(len(labels)-1,-1,-1):
                if i==j :
                    continue 
                intersection=cv2.bitwise_and(expend_mask(masks[i].copy(),thickness=36), expend_mask(masks[j].copy(),thickness=36))
                if cv2.countNonZero(intersection) > 0:
                    neibor_list.append(labels[j])

            if len(list(set(neighbor_relation[key]) & set(neibor_list)))==0:
                id_alloy_wheel=i
                if 'tyre' in key:
                    for m in range(len(labels)-1,-1,-1):
                        if 'alloy_wheel' not in labels[m]:
                            continue
                        box1 = torch.tensor([bboxes[i]], dtype=torch.float)
                        box2 = torch.tensor([bboxes[m]], dtype=torch.float)
                        iou = bops.box_iou(box1, box2)
                        if iou[0][0] > 0 :
                            id_alloy_wheel = m
                            break
                if id_alloy_wheel != i:
                    labels.pop(max(i,id_alloy_wheel))
                    scores.pop(max(i,id_alloy_wheel))
                    masks.pop(max(i,id_alloy_wheel))
                    bboxes.pop(max(i,id_alloy_wheel)) 
                        
                labels.pop(min(i,id_alloy_wheel))
                scores.pop(min(i,id_alloy_wheel))
                masks.pop(min(i,id_alloy_wheel))
                bboxes.pop(min(i,id_alloy_wheel))    

    # print('mmmmmmmmmmmmmmmmmm',labels)
    return labels,masks,scores,bboxes


def remove_carpart_from_view(labels, masks, scores,bboxes):
    front_num=0
    back_num =0 
    for label in labels:
        if 'f' in label.split('+')[-1]:
            front_num +=1 
        else:
            back_num +=1

    for i in range(len(labels) - 1, 0, -1):
        for j in range(len(labels) - 1, -1, -1):
            if 'qpa_quarter_panel' in labels[i] and 'fender' in labels[j]:
                if front_num  >back_num :
                    labels.pop(i)
                    scores.pop(i)
                    masks.pop(i)
                    bboxes.pop(i)
                    break

            if 'fender' in labels[i] and 'qpa_quarter_panel' in labels[j]:
                if front_num  < back_num :
                    labels.pop(i)
                    scores.pop(i)
                    masks.pop(i)
                    bboxes.pop(i)
                    break

unique_side=['hli_head_light','fender']
def remove_wrong_unique_side_carpart(label2center, masks, scores,bboxes):
    
    for unique_capart in unique_side:
        unique_ids=[i for i,carpart in enumerate(label2center) if carpart[0]==unique_capart]

        grille_y_min=10000
        grille_id =-1
        for j,carpart in enumerate(label2center):
            if carpart[0]=='grille' and carpart[2] < grille_y_min:
                grille_y_min=carpart[2]
                grille_id=j

        if len(unique_ids) > 1 and grille_id  >0:
            for i  in range(len(unique_ids)-1,0,-1):
                i_id= unique_ids[i]
                for j in range(i-1,-1,-1):
                    j_id= unique_ids[j]
                    i_unique_x= label2center[i_id][1]
                    j_unique_x= label2center[j_id][1]
                    grille_x=label2center[grille_id][1]

                    if (i_unique_x- grille_x)* (j_unique_x- grille_x) >0 :
                        if abs(i_unique_x- grille_x) < abs(j_unique_x-grille_x):
                            label2center[j_id]=label2center[i_id]
                            masks[j_id] =masks[i_id]
                            scores[j_id] =scores[i_id]
                            bboxes[j_id] =bboxes[i_id]
                    
                        label2center.pop(i_id)
                        masks.pop(i_id)
                        scores.pop(i_id)
                        bboxes.pop(i_id)
                        grille_id=grille_id -1


def check_mask_inside_missing(carpart_m,missing_m):
    if carpart_m is None or missing_m is None:
        return False
    intersection = np.logical_and(carpart_m, missing_m)
    area_intersection=np.sum(intersection)
    area_mask_i=np.sum(carpart_m)
    area_mask_j = np.sum(missing_m)
    if area_intersection/ area_mask_i > 0.8:
        return True 
    return False 

def filter_by_missing(pred_json):
    if 'missing' not in pred_json:
        return 
    carpart_pred=pred_json['carpart']

    for i in range(len(carpart_pred['labels'])-1,-1,-1):
        in_mising=False 
        for mising_mask in pred_json['missing']['masks']:
            if check_mask_inside_missing(carpart_pred['masks'][i],mising_mask):
                in_mising=True
                break
        
        if in_mising:
            carpart_pred['labels'].pop(i)
            carpart_pred['scores'].pop(i)
            carpart_pred['masks'].pop(i)
            carpart_pred['bboxes'].pop(i)
    
def filter_carpart(pred_json):
    carpart_pred=pred_json['carpart']
    # print('debug filter carpart ***')

    for correct_masks_label in correct_masks_labels:
        # print('debug : ',correct_masks_label)
        correct_masks(carpart_pred['labels'],carpart_pred['masks'],correct_masks_label)
    carpart_pred['labels']=correct_font_back_labels(carpart_pred['labels'],carpart_pred['masks'])
    merge_carpart(carpart_pred['labels'],carpart_pred['masks'],carpart_pred['scores'],carpart_pred['bboxes'])
    filter_by_missing(pred_json)
    correct_labels(pred_json)

def correct_quarter_panel_base_fuel_tank_door(pred_json):
    labels = pred_json['carpart']['labels']
    masks = pred_json['carpart']['masks']

    # if check_carpart_in_list('fuel_tank_door',labels) : 
    for idx, label in enumerate(labels):
        if label == 'fuel_tank_door':
            x1,y1,x2,y2 = pred_json['carpart']['bboxes'][idx]
            center = (int((y1+y2)/2),int((x1+x2)/2))

            for idy,mask in enumerate(masks):
                if mask[center] and labels[idy]=='fender+rf':
                    pred_json['carpart']['labels'][idy]='qpa_quarter_panel+lb'
                    print('change fender to quarter panel yolo !!')
                    return pred_json
                elif mask[center] and labels[idy]=='fender+lf':
                    pred_json['carpart']['labels'][idy]='qpa_quarter_panel+rb'
                    print('change fender to quarter panel yolo !!')
                    return pred_json

    return pred_json

def check_one_door_car(labels):
    labels=[label1.split('+')[0] for label1 in labels]

    if 'fender' in labels and 'qpa_quarter_panel' in labels: 
        if labels.count('door')==1:
            return True, False
        if labels.count('door')> 1:
            return False,True 
    return False,False

def adjust_label_on_sigle_door_car(pred_jsons):
    for pred_json in pred_jsons:
        labels=pred_json['carpart']['labels']

        labels_not_index=[label.split(':')[0] for label in labels]
        if ('door+lb' in labels_not_index  or 'door+rb' in labels_not_index) and ('qpa_quarter_panel+lb' in labels_not_index  or 'qpa_quarter_panel+rb' in labels_not_index):

            for i, label in enumerate(labels):
                if 'door+lb' in label or  'door+rb' in label:
                    labels[i]=labels[i].replace('b','f')

        if ('door+lf' in labels_not_index  or 'door+rf' in labels_not_index) and ('door+lb' in labels_not_index  or 'door+rb' in labels_not_index) :
            for i, label in enumerate(labels):
                if 'door+lb' in label or  'door+rb' in label:
                    labels[i]=labels[i].replace('door','qpa_quarter_panel')

    for pred_json in pred_jsons:
        labels=pred_json['carpart']['labels']

    return pred_jsons
