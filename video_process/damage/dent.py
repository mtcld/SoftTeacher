import cv2
import numpy as np
from video_process.utils.util import is_eligible_damage_on_part
from video_process.damage import DamageDecorator,FilterDamage,FilterDecorator
from video_process.utils.mask import expend_mask
import torch 
import math 

import os 
import torchvision.ops.boxes as bops

from video_process import yolo_models
# from car_damage.config.cfg import envs
# from video_process.utils.visualize import visual_yolo_result


def check_by_view(dent_index,pred_json):
    carpart_info= pred_json['carpart']

    score = pred_json["dent"]["scores"][dent_index]

    pred_damage = pred_json["dent"]["masks"][dent_index]
    for cat,carpart_mask,carpart_score in zip(carpart_info['labels'],carpart_info["masks"],carpart_info['scores']):

        carpart_area=cv2.countNonZero(carpart_mask)
        overlap_area = cv2.bitwise_and(carpart_mask, pred_damage)
        damage_pixel=cv2.countNonZero(overlap_area)

        if damage_pixel / min(cv2.countNonZero(pred_damage),carpart_area) < 0.02:
            continue

        if ('fender' in cat or 'qpa_quarter_panel' in cat):
            return True 
    return False

class RemoveDentByHandle(FilterDecorator):
    def __call__(self,image,pred_json):
        carpart_info=pred_json['carpart']
        for i in range(len(pred_json["dent"]["masks"])-1,-1,-1):
            for cat,carpart_mask in zip(carpart_info['labels'], carpart_info["masks"]):

                if 'handle' not in cat:
                    continue
                carpart_mask = expend_mask(carpart_mask)
                damage_mask=np.uint8(pred_json["dent"]["masks"][i])

                overlap_area = cv2.bitwise_and(carpart_mask, damage_mask)
                over_pixel = cv2.countNonZero(overlap_area)

                if over_pixel/cv2.countNonZero(damage_mask) > 0.3:
                    pred_json["dent"]["masks"].pop(i)
                    pred_json["dent"]["bboxes"].pop(i)
                    pred_json["dent"]["scores"].pop(i)
                    break

        return self.component(image,pred_json)

class RemoveDentByYolo(FilterDecorator):    
    def __call__(self,image,pred_json):
        boxes,confs,_=yolo_models['dent'](image)
        # print('debug yolo : ',boxes)
        # if envs.WHICH=='DEV-LOCAL':
        #     visual_yolo_result(image,boxes,confs,'dent')

        # for idx,mask in enumerate(pred_json['dent']['masks']):
        #     mask = mask.astype(bool)
        #     image[mask] = 0.5*image[mask] + 0.5*np.array([0,255,0])
        #     cv2.putText(image, 'cbn '+str(round(pred_json['dent']['scores'][idx],2)), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)

        # for bbox,conf  in zip(boxes,confs):
        #     x1,y1,x2,y2=bbox
        #     cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),1)
        #     cv2.putText(image, 'yolo'+str(round(conf.item(),2)), (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)
    
        
            
        keep_masks=[]
        for i,mask_i in enumerate(pred_json["dent"]["masks"]):
            area_i=cv2.countNonZero(mask_i)
            for j,mask_j in enumerate(pred_json["dent"]["masks"]):
                area_j=cv2.countNonZero(mask_j)
                if i ==j :
                    continue 
                overlap=cv2.bitwise_and(mask_i,mask_j)

                if cv2.countNonZero(overlap)/max(min(area_i,area_j),1) >0.1:
                    keep_masks.append(i)
                    break

        roi=None
        for i in range(len(pred_json["dent"]["bboxes"])-1,-1,-1):
            score= pred_json["dent"]["scores"][i]
            if score <0.55:  
                # if (not check_by_dent_cb(i,pred_json)) or (i not in keep_masks):
                if (i not in keep_masks):
                    pred_json["dent"]["masks"].pop(i)
                    pred_json["dent"]["bboxes"].pop(i)
                    pred_json["dent"]["scores"].pop(i)
                    continue 

            overlap=False
            yolo_cof= -1
            for yolo_c,rec in zip(confs,boxes):
                if roi is not None:
                    rec[0] =rec[0] + roi[0]
                    rec[1] =rec[1] + roi[1]
                    rec[2] =rec[2] + roi[0]
                    rec[3] =rec[3] + roi[1]
                box1 = torch.tensor([pred_json["dent"]["bboxes"][i]], dtype=torch.float)
                box2 = torch.tensor([rec], dtype=torch.float)
                iou = bops.box_iou(box1, box2)
                if iou[0][0] > 0.05 :
                    overlap=True
                    yolo_cof= yolo_c
                    break
            # cv2.putText(image, 'yolo | '+str(overlap)+' | '+str(yolo_cof)+' | '+str(check_by_view(i,pred_json)), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)
            
            if not overlap or (overlap and check_by_view(i,pred_json) and yolo_cof<0.4):
                pred_json["dent"]["masks"].pop(i)
                pred_json["dent"]["bboxes"].pop(i)
                pred_json["dent"]["scores"].pop(i)
        
        # cv2.imwrite('debug_yolo.jpg',image)
        
        return self.component(image,pred_json)


filter=FilterDamage()
filter=RemoveDentByYolo(filter)
filter=RemoveDentByHandle(filter)

class Dent(DamageDecorator):
    def get_damage2carpart(self,image,pred_json,final_output):
        carpart_info= pred_json['carpart']
        if carpart_info['view'] is None:
            return self.component.get_damage2carpart(image,pred_json,final_output)
        pred_json=filter(image,pred_json)
        
        for id_carpart,(cat,carpart_mask) in enumerate(zip(carpart_info['labels'],carpart_info["masks"])):
            carpart_area= cv2.countNonZero(carpart_mask)
            if carpart_area/(carpart_info['car_area']) <0.01:
                continue 

            max_score = 0
            dam='dent'
            if not is_eligible_damage_on_part(cat, dam):
                continue
            damage_res, pred_damage_list,score_list=pred_json[dam]["labels"],pred_json[dam]["masks"],pred_json[dam]["scores"]
        
            ndamage_per_carpart=0
            for kn in range(len(pred_damage_list)):
                pred_damage = pred_damage_list[kn]

                overlap_area = cv2.bitwise_and(carpart_mask, pred_damage)
                damage_pixel=cv2.countNonZero(overlap_area)
                if damage_pixel > 0:

                    if damage_pixel / min(cv2.countNonZero(pred_damage),carpart_area) < 0.17:
                        continue
                    ndamage_per_carpart +=1
                    # max_score = max(max_score, float(score_list[kn]))

                    damage_score = [dam, score_list[kn],kn,id_carpart,False]

                    # if max_score > 0:
                    if cat not in final_output.keys():
                        final_output[cat] = []
                        final_output[cat].append(damage_score)
                    else:
                        final_output[cat].append(damage_score)

        return self.component.get_damage2carpart(image,pred_json,final_output)