# from video_process.car.view import Car_View
import enum

from mmdet.core.bbox.assigners import HungarianAssigner
from video_process.carpart.segment import CarpartInfo
# from video_process.damage import Damage
# from video_process.damage.scratch import Scratch
# from video_process.damage.dent import Dent
from video_process.matching import estimate_position

import numpy as np
import cv2
# import matplotlib.pyplot as plt
import json
# import copy
# import random
import pandas as pd
from pathlib import Path 
import os
import glob
import math as m
# import dlib 
import torch 
from scipy.optimize import linear_sum_assignment

from video_process.Hungarian_matching import Hungarian


from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector as st_init_detector


cp_model = init_detector('checkpoint/carpart_rear_exp_2/carpart_rear.py',
                      'checkpoint/carpart_rear_exp_2/epoch_29.pth',device='cuda:0')

car_model = init_detector('thirdparty/mmdetection/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
                          'checkpoint/coco_pretrain/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth',device='cuda:0')


# scratch_model = init_detector('checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/swa-scratch-copy-paste-HSV-LSJ.py',
#                         'checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/best_bbox_mAP.pth',device='cuda:0')

# dent_model = st_init_detector('checkpoint/19_02_2022/dent/dent_mask.py',
#                         'checkpoint/19_02_2022/dent/iter_207000.pth',device='cuda:0')

# crack_model = init_detector('checkpoint/origin_crack_cp_exp_3/copy-paste-with-crop-images.py',
#                         'checkpoint/origin_crack_cp_exp_3/epoch_29.pth',device='cuda:2')

models = {'car':car_model,'carpart':cp_model}

carpart_if=CarpartInfo()

center = (419,284)
radius = 252

def model_inference(model, image,score_thre=0.5):
    result = inference_detector(model,image)
    img_,pred_boxes,pred_segms,pred_labels,pred_scores = show_result_pyplot(model, image.copy(), result,score_thr=score_thre)
    
    if len(pred_boxes) == 0:
        return {'labels':[],'scores':[],'masks':[],'bboxes':[]}
    
    out_pred = []
    for pred in pred_boxes:
        box = [p for b in pred for p in b]
        out_pred.append(box)
    return {"labels": [model.CLASSES[l] for l in pred_labels], "scores": pred_scores.tolist(), "masks": [s.astype(np.uint8) for s in pred_segms],'bboxes':out_pred}

def car_model_inference(model, image):
    ## car inference 
    result = inference_detector(model,image)
    img_,pred_boxes,pred_segms,pred_labels,pred_scores = show_result_pyplot(model, image.copy(), result,score_thr=0.5)
    
    index = []
    for l in pred_labels:
        if model.CLASSES[l] in ['car','truck']:
            index.append(True)
        else:
            index.append(False)
    
    pred_labels = pred_labels[index]
    pred_boxes = np.array(pred_boxes)[index].tolist()
    pred_scores = pred_scores[index]
    pred_segms = pred_segms[index]

    out_pred = []
    for pred in pred_boxes:
        box = [p for b in pred for p in b]
        out_pred.append(box)
    # print('debug car inference  : ',pred_labels)
    return {"labels": [model.CLASSES[l] for l in pred_labels], "scores": pred_scores.tolist(), "masks": [s.astype(np.uint8) for s in pred_segms],'bboxes':out_pred}

def damage_model_inference(model,pred_json,image,score_thre = 0.5):
    result = inference_detector(model,image)
    img_, pred_boxes, pred_segms, pred_labels, pred_scores = show_result_pyplot(model,image.copy(),result, score_thr=score_thre)
    
    if len(pred_boxes) == 0:
        pred_json[0][model.CLASSES[0]] = {'labels':[],'scores':[],'masks':[],'bboxes':[]}

        return pred_json
    
    out_pred = []
    for pred in pred_boxes:
        box = [p for b in pred for p in b]
        out_pred.append(box)

    pred_json[0][model.CLASSES[0]] = {"labels": [model.CLASSES[l] for l in pred_labels], "scores": pred_scores.tolist(), "masks": [s.astype(np.uint8) for s in pred_segms],'bboxes':out_pred}

    return pred_json

def damage_model_inference_only(image):
    # out = {}
    scratch_result = model_inference(scratch_model,image,0.5)
    dent_result = model_inference(dent_model,image,0.5)
    # crack_result = model_inference(crack_model,image,0.5)

    # return {'scratch':scratch_result,'dent':dent_result,'crack':crack_result}
    return {'dent':dent_result,'scratch':scratch_result}

def get_model_prediction(image):
    out = {}
    out['car'] = car_model_inference(models['car'],image)
    out['carpart'] = model_inference(models['carpart'],image)
    
    return [out]

def compare_masks(image,damage,pred_json):
    final_output = {}

    carpart_info=pred_json['carpart']
    if len(carpart_info['labels'])==0 or  (len(carpart_info['labels'])>2 and  ('c_90_90' in carpart_info['totaled_info']['view_img']  or 'c_90_270' in carpart_info['totaled_info']['view_img'])):
        return pred_json,final_output

    pred_json,final_output=damage.get_damage2carpart(image,pred_json,final_output)

    
    rm_part_list = [carpart for carpart, value in final_output.items() if (
                "tyre" in carpart or "alloy_wheel" in carpart or "handle" in carpart)]
    for rm_part in rm_part_list:
        if rm_part in final_output:
            del final_output[rm_part]

    return pred_json,final_output

class EMA:
    def __init__(self,values=[],span=2):
        self.df = pd.DataFrame({'values':values})
        self.default_span = span
        self.current_span = span
        self.activate_new_span = False
        self.gap = 90
        self.ema_list = self.df.ewm(span=self.current_span).mean()['values'].tolist()

    def add(self,value):
        value = int(value)
        if len(self.df) == 0:
            self.df.loc[len(self.df)] = [value]
            self.ema_list.append(value)
            return value

        if abs(value - self.df.loc[len(self.df)-1]['values']) > self.gap:
            self.current_span = 0
            self.activate_new_span = True

        self.df.loc[len(self.df)] = [value]

        if self.current_span < self.default_span :
            if self.activate_new_span is True and self.current_span == 1 :
                self.activate_new_span = False
            else:
                self.current_span += 1

        ema_value = self.df.ewm(span=self.current_span).mean()['values'].tolist()[-1]
        self.ema_list.append(ema_value)
        
        return ema_value 
    
    def get_origin_values(self):
        return self.df['values'].tolist()
    
    def get_ema_values(self):
        return self.ema_list

def draw_icon(image,start_angle,end_angle):
    if start_angle > 180 : 
        start_angle = start_angle - 360
    
    if end_angle > 180:
        end_angle = end_angle - 360
    
    if start_angle < -90 : 
        start_angle = start_angle + 360
    
    if end_angle < -90:
        end_angle = end_angle + 360

    if abs(end_angle - start_angle) > 300:
        end_angle = end_angle + 360
        # ns = min(start_angle,end_angle)
        # ne = max(start_angle,end_angle)
        # ns = ns + 360

        # start_angle = ns
        # end_angle = ne

    print('debug loading angle : ',start_angle,end_angle,' | ',start_angle-90,end_angle-90)
    image = cv2.ellipse(image,center,(radius,radius),0,start_angle-90,end_angle-90,(255,255,0),-1)

    return image

def outlier(a,b,c):
    a1 = m.atan(1/(b-a)) / m.pi*180
    a2 = m.atan(1/(b-c)) / m.pi*180
    
    if a1 < 0 : 
        a1 = 180 + a1
    if a2 < 0 :
        a2 = 180 + a2
    
    angle =  (a1+a2)
    
    check = False
    if angle < 3:
        check = True
        if a < 5 or c < 5 :
            check = False
    
    if angle > 357 :
        check = True
        if b < 10 :
            check = False
    
    return check

def clean_outlier(pred_json,bin_length):
    angles = [int(k)*bin_length for k in pred_json.keys()]

    for i in range(len(angles)):
        if i == 0 or i == len(angles)-1 :
            continue
        
        if outlier(angles[i-1],angles[i],angles[i+1]):
            pred_json.pop(str(angles[i]//bin_length),None)

    return pred_json

def normalize(box,W,H):
    xyxy = np.array([box[0],box[2]]).reshape(-1)
    box = [xyxy[0]/W,xyxy[1]/H,xyxy[2]/W,xyxy[3]/H]
    return torch.tensor(box)


def iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
):

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
    iou_l = iou_loss(box1,box2)
    l1_l = l1_loss(box1,box2)
    
    label_l = label_loss(label1,label2)
    
    return iou_l+l1_l+label_l

def bipartite_matching(track_info, detect_info,pred_json,W,H):
    # base on tracking of previous frame + detect of current frame -> relabel current detection -> output new roi list and pred_json
    track_labels = []
    track_boxes = []
    # track_ind = []

    for label,box,idx in track_info:
        track_labels.append(label)
        track_boxes.append(normalize(box,W,H))
        # track_ind.append(idx)
    
    detect_labels = []
    detect_boxes = []
    detect_ind = []

    for label, box, idx in detect_info:
        detect_labels.append(label)
        detect_boxes.append(normalize(box,W,H))
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

    # return detect_labels[row_ind],track_labels[col_ind]

def main():
    
    bin_length = 5

    # damage = Damage()
    # damage = Dent(damage)
    # damage = Scratch(damage)
    # record_flag = False
    # record_length = 0
    tracking_flag = False 
    tracking_frame = 0
    # trackers = cv2.legacy.MultiTracker_create()
    trackers_list = []
    


    damage_clrs = {'scratch':np.array([255,0,0]),'dent':np.array([0,255,0]),'crack':np.array([0,0,255])} 

    # video_files = glob.glob('video/*.MOV')
    video_files = glob.glob('video_tool/demo.avi')
    # video_files = ['video/video_2/FBAI_Car_05.MOV','video/20220504_142734.mp4']
    out_path = 'video_out'
    for file in video_files:
        data_frame = {}
        # cap = cv2.VideoCapture('video/VID_20220709_122339.mp4')
        icon = cv2.imread('video_process/icon_car.png')
        icon = cv2.circle(icon.copy(),center,radius,(255,0,0),10)
        
        name = os.path.basename(file)
        name = name[:name.rfind('.')]

        # images_video_path = out_path+'/'+name+'_bin_'+str(bin_length)+'_first_frame'
        # Path(images_video_path).mkdir(parents=True,exist_ok=True)

        cap = cv2.VideoCapture(file)
        frame_w = int(cap.get(3))
        frame_h = int(cap.get(4))

        hungarian = Hungarian(frame_w,frame_h)

        print('debug wh : ',frame_w,frame_h)

        # video_writer = cv2.VideoWriter(out_path+'/'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),24,(frame_w*2,frame_h*2)) 
        video_writer = cv2.VideoWriter('video_tool/demo_out_glue.avi',cv2.VideoWriter_fourcc('M','J','P','G'),5,(frame_w,frame_h))
        count = 0
        # views = EMA(span=15)

        # pre_v = 0
        # curr_v = 0

        # damages_by_frame = {}
        # damaged_carpart = {}

        roi_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret == True :             
                if count % 1 == 0:
                    tracking_info = []
                    if tracking_flag : 
                        # success,boxes =trackers.update(frame)
                        # for tracker in trackers_list:
                        #     tracker.update(frame)
                        #     pos = tracker.get_position()
                            # cv2.rectangle(frame,(int(pos.left()),int(pos.top())),(int(pos.right()),int(pos.bottom())),(0,0,255),1)
                        # print(success,boxes)
                        # if success : 
                        # for idx,b in enumerate(boxes):
                            # cv2.rectangle(frame,(int(b[0]),int(b[1])),(int(b[2]+b[0]),int(b[3]+b[1])),(0,idx*(255//len(result[0]['carpart']['bboxes'])),255),1)
                        
                        for roi_data in roi_list:
                            dst = estimate_position(pre_frame,frame,roi_data[1])
                            
                            frame  = cv2.polylines(frame,[np.int32(dst)],True,[255,0,0],2,cv2.LINE_AA)
                            p = np.int32(dst)[2][0]
                            cv2.putText(frame,roi_data[0],(p[0]-70,p[1]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.6,color=(0,0,255),thickness=2)
                            
                            roi_info = [roi_data[0],np.array(dst).reshape(-1,2).tolist(),roi_data[2]]
                            tracking_info.append(roi_info)
                            # print('debug dst : ',dst)
                        data_frame[count] = {'track':tracking_info,'detect':[]}
                        pre_frame = frame.copy()
                        tracking_frame += 1

                        if tracking_frame % 5 == 0:
                            tracking_flag = False
                            # trackers = cv2.legacy.MultiTracker_create()

                    if tracking_flag == False:
                        result = get_model_prediction(frame)
                        print('frame time : ',count)
                        # trackers_list = []
                        roi_list = []
                        result = carpart_if.add_carpart_info(['krug/'],result)

                        # detect_info = []
                        for idx,b in enumerate(result[0]['carpart']['bboxes']):
                            # cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,0,255),1)
                            # cv2.putText(frame,result[0]['carpart']['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,idx*(255//len(result[0]['carpart']['bboxes'])),255),1)
                            # print(result[0]['carpart']['labels'][idx])
                            # if result[0]['carpart']['labels'][idx] in ['handle','sli_side_turn_light+lf']:
                            
                            if any([i in result[0]['carpart']['labels'][idx] for i in ['door','fender']]):
                                cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,255,255),1)
                                # score = round(result[0]['carpart']['scores'][idx],2)
                                cv2.putText(frame,result[0]['carpart']['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,0,255),thickness=2)

                                roi = [[int(b[0]),int(b[1])],[int(b[2]),int(b[1])],[int(b[2]),int(b[3])],[int(b[0]),int(b[3])]]
                                roi_list.append([result[0]['carpart']['labels'][idx],roi,idx])
                                pre_frame = frame.copy()
                                tracking_flag = True
                                
                                # roi_info = [result[0]['carpart']['labels'][idx],roi,idx]
                                # detect_info.append(roi_info)
                        
                        roi_list, result = hungarian.bipartite_matching(tracking_info,roi_list,result)
                        # roi_list,result = bipartite_matching(tracking_info,roi_list,result,frame_w,frame_h)

                        data_frame[count] = {'track':[],'detect':roi_list}

                                # break
                            # trac
                    
                    # print('yolll : ')
                    # print(result[0]['carpart']['bboxes'])
                    # print(result[0]['carpart']['labels'])

                    # if result[0]['carpart']['view'] is None:
                    #     continue

                    # if str(result[0]['carpart']['view']) == '300':
                    #     record_flag = True

                    # cv2.putText(frame,str(result[0]['carpart']['view']),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                    # for idx,b in enumerate(result[0]['carpart']['bboxes']):
                    #     cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,0,255),1)
                    #     cv2.putText(frame,result[0]['carpart']['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)


                    # print('debug v:',v)
                    # if result[0]['carpart']['view'] is None:
                    #     # print('alo')
                    #     continue
                    

                    # v = views.add(result[0]['carpart']['view'])
                    # v = int(v)
                    # curr_v = v
                    # if count == 0:
                    #     pre_v = v
                    
                    # icon = draw_icon(icon,pre_v,curr_v)
                    # icon_frame = cv2.resize(icon.copy(),(frame_w,frame_h))
                    # pre_v = curr_v
                    # if curr_v // bin_length not in damages_by_frame.keys() :
                    #     cv2.imwrite(images_video_path+'/'+'bin_'+str(curr_v // bin_length)+'_frame_'+str(curr_v)+'.jpg',frame)

                    #     result = damage_model_inference(dent_model,result,frame,score_thre=0.35)
                    #     result = damage_model_inference(scratch_model,result,frame,score_thre=0.6)

                    #     pred_result,damage_result = compare_masks(frame,damage,result[0])

                    #     print('debug final output : ',damage_result)
                    #     for k in damage_result.keys():
                    #         if k in damaged_carpart.keys():
                    #             damaged_carpart[k] += 1
                    #         else:
                    #             damaged_carpart[k] = 1

                    #     damages_by_frame[curr_v // bin_length] = damage_result
                    #     print('debug damages_by_frame : ',damages_by_frame)

                    #     # draw scratch
                    #     damages_frame = frame.copy()
                    #     for mask in pred_result['scratch']['masks']:
                    #         mask = mask.astype(bool)
                    #         damages_frame[mask] = damages_frame[mask]*0.5+damage_clrs['scratch']*0.5
                        
                    #     for mask in pred_result['dent']['masks']:
                    #         mask = mask.astype(bool)
                    #         damages_frame[mask] = damages_frame[mask]*0.5+damage_clrs['dent']*0.5

                    ## draw main car and its damaged car parts
                    # carpart_frame = frame.copy()
                    # for carpart, confirmation in damaged_carpart.items():
                    #     if carpart in result[0]['carpart']['labels']:
                    #         index = result[0]['carpart']['labels'].index(carpart)
                    #         mask = result[0]['carpart']['masks'][index]
                    #         mask = mask.astype(bool)
                    #         if confirmation == 1:
                    #             clr = np.array([0,255,0])
                    #         else:
                    #             clr = np.array([0,0,255])

                    #         carpart_frame[mask] = carpart_frame[mask]*0.5 + clr*0.5

                    # main_car = result[0]['car']['main_car']

                    # cv2.rectangle(frame,(main_car[0],main_car[1]),(main_car[0]+ main_car[2],main_car[1]+main_car[3]),(255,0,0),5)

                    # frame = cv2.putText(frame,str(result[0]['carpart']['view'])+ ' | '+str(v),(main_car[0],main_car[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                    # cv2.imwrite('debug_icon.jpg',icon)
                    
                    # output_frame = np.hstack((np.vstack((frame,carpart_frame)),np.vstack((damages_frame,icon_frame))))
                    # cv2.imwrite('debug_view.jpg',output_frame)
                    cv2.imwrite('video_tool/debug_view.jpg',frame)
                    video_writer.write(frame)
                #     if record_flag : 
                #         if record_length < 40:
                #             video_writer.write(frame)
                #             record_length += 1
                
                # if record_length == 40 :
                #     break
                count += 1
            else:
                break

        # with open(out_path+'/'+name+'_damges_by_first_frame_in_bin.json', 'w', encoding='utf-8') as f:
        #     json.dump(damages_by_frame, f, ensure_ascii=False, indent=4)
        with open('video_tool/track_detect_frame_info.json', 'w', encoding='utf-8') as f:
            json.dump(data_frame, f, ensure_ascii=False, indent=4)


    ############################################################### damage models inference only##############################################################
    # video_path = Path('video')
    # out_path = Path('video_out')
    # out_path.mkdir(parents=True, exist_ok=True)
    # 

    # video_files = [os.path.basename(f) for f in video_path.iterdir() if f.is_file()]
    # out_video_files = [os.path.basename(f) for f in out_path.iterdir() if f.is_file()]
    # out_video_names = [f[:f.rfind('.')] for f in out_video_files]

    # for f in video_files:
    #     name = f[:f.rfind('.')]
    #     # if name in out_video_names:
    #     #     continue
        
    #     print(name)
    #     cap = cv2.VideoCapture(str(video_path/f))
    #     frame_w = int(cap.get(3))
    #     frame_h = int(cap.get(4))

        
    #     video_writer = cv2.VideoWriter(str(out_path)+'/'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),24,(frame_w,frame_h)) 
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         frame_count = 0
    #         if ret == True:
    #             if frame_count % 2 == 0:
    #                 result = damage_model_inference_only(frame)
    #                 # print(result)
    #                 for k in result.keys():
    #                     for mask in result[k]['masks']:
    #                         mask = mask.astype(bool)
    #                         frame[mask] = frame[mask]*0.5 + clrs[k]*0.5
    #                 cv2.imwrite('debug_view.jpg',frame)
    #                 video_writer.write(frame)
    #             frame_count += 1
    #             # pass 
    #         else:
    #             break

if __name__=='__main__':
    main()