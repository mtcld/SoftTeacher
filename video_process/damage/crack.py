import cv2
import numpy as np
from car_damage.postprocess.utils.mask import expend_mask,create_id_mask,create_mask
from car_damage.postprocess.damage.loose import reject_loose
from car_damage.postprocess.damage import DamageDecorator,FilterDamage,FilterDecorator

import cv2
import numpy as np
import torch 
import os 
import torchvision.ops.boxes as bops
from car_damage.postprocess.models import yolo_models
from car_damage.config.cfg import envs
from car_damage.postprocess.utils.visualize import visual_yolo_result


class RemoveCrackByLoose(FilterDecorator):

    def __call__(self,image_path,pred_json):
        if 'loose' not in pred_json:
            return self.component(pred_json)
            
        for i in range(len(pred_json["bbox_crack"]["masks"])-1,-1,-1):
            for loose_mask in pred_json["loose"]["masks"]:
                overlap_area = cv2.bitwise_and(loose_mask,pred_json["bbox_crack"]["masks"][i]*255)
                over_pixel = cv2.countNonZero(overlap_area)
                crack_pixel = cv2.countNonZero(pred_json["bbox_crack"]["masks"][i])
                loos_pixel = cv2.countNonZero(loose_mask)

                if max(crack_pixel,loos_pixel)==0 :
                    pred_json["bbox_crack"]["masks"].pop(i)
                    pred_json["bbox_crack"]["bboxes"].pop(i)
                    break  
                if loos_pixel==0:
                    continue
                if over_pixel/loos_pixel > 0.5:
                    pred_json["bbox_crack"]["masks"].pop(i)
                    pred_json["bbox_crack"]["bboxes"].pop(i)
                    break

        return self.component(image_path,pred_json)

class RemoveCrackByMissing(FilterDecorator):
    def __call__(self,image_path,pred_json):
        if 'missing' not in pred_json:
            return self.component(pred_json)
        for i in range(len(pred_json["bbox_crack"]["masks"])-1,-1,-1):
            for missing_mask in pred_json["missing"]["masks"]:
                overlap_area = cv2.bitwise_and(missing_mask,pred_json["bbox_crack"]["masks"][i]*255)

                over_pixel = cv2.countNonZero(overlap_area)
                crack_pixel = cv2.countNonZero(pred_json["bbox_crack"]["masks"][i])
                missing_pixel = cv2.countNonZero(missing_mask)

                if max(crack_pixel,missing_pixel)==0 :
                    pred_json["bbox_crack"]["masks"].pop(i)
                    pred_json["bbox_crack"]["bboxes"].pop(i)
                    break  
                
                if over_pixel/missing_pixel > 0.5:
                    pred_json["bbox_crack"]["masks"].pop(i)
                    pred_json["bbox_crack"]["bboxes"].pop(i)
                    break
        return self.component(image_path,pred_json)


class RemoveCrackByScratch(FilterDecorator):

    def __call__(self,image_path,pred_json):
        for i in range(len(pred_json["bbox_crack"]["bboxes"])-1,-1,-1):
            
            crack_bbox=pred_json["bbox_crack"]["bboxes"][i]
            # scr_bboxs,confs,_ =yolo_models['scratch'](image_path) 
            # print(scr_bboxs)
            for scr_bbox,cof in zip(pred_json['scratch']['bboxes'],pred_json['scratch']['scores']):
                
                # if cof.item() <0.6:
                #     continue 
                box1 = torch.tensor([scr_bbox], dtype=torch.float)
                box2 = torch.tensor([crack_bbox], dtype=torch.float)
                iou = bops.box_iou(box1, box2)
                if iou > 0.7:
                    pred_json["bbox_crack"]["masks"].pop(i)
                    pred_json["bbox_crack"]["bboxes"].pop(i)
                    break

        return self.component(image_path,pred_json)


filter= FilterDamage()
filter=RemoveCrackByLoose(filter)
# filter=RemoveCrackByMissing(filter)
filter=RemoveCrackByScratch(filter)

bbox_Yolo=yolo_models['crack'] 

class Crack(DamageDecorator):
    def get_damage(self,image_path,pred_json,final_output):
        bboxes,confs,shape = bbox_Yolo(image_path)

        if envs.WHICH=='DEV-LOCAL':
            visual_yolo_result(image_path,bboxes,confs,'crack')
            
        if len(bboxes)==0:
            return pred_json,final_output
        
        masks=[create_mask([[np.array([[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]).reshape(-1,1,2)]],\
             shape[1],shape[0]) for bbox in bboxes]
        pred_json['bbox_crack']={'masks':masks,'bboxes':bboxes}
        
        pred_json=filter(image_path,pred_json)
        if len(pred_json['bbox_crack']['bboxes'])==0:
            return pred_json,final_output
        carpart_info= pred_json['carpart'].copy()
        carpart_labels= carpart_info['labels'].copy()
        carpart_contours=carpart_info["contours"].copy()
        if len(carpart_labels)<2:
            return pred_json,final_output
        
        for i in range(len(carpart_info['scores'])-1,-1,-1):
            if carpart_info['scores'][i] <0.7 or 'tyre' in carpart_info['labels'][i]:
                carpart_labels.pop(i)
                carpart_contours.pop(i)

        id_mask=create_id_mask(carpart_contours,carpart_info["masks"][0].shape[:2])

        for bbox,d_score in zip(pred_json['bbox_crack']['bboxes'],confs):       
            checked_mask=False 
            for pred_damage,bbox1  in zip(pred_json['crack']['masks'],pred_json['crack']['bboxes']):
                box1 = torch.tensor([bbox], dtype=torch.float)
                box2 = torch.tensor([bbox1], dtype=torch.float)
                iou = bops.box_iou(box1, box2)
                if iou > 0.4:                    
                    damage_pixel=-1
                    for carpart,carpart_mask,score in zip(carpart_info['labels'],carpart_info["masks"],carpart_info["scores"]):

                        if score <0.7 or ('door' in carpart  and carpart_info['view'] is None) or  'tyre' in carpart:
                            continue 
                        
                        if len(carpart_info['labels']) >13 and 130< int(carpart_info['view'])< 240 and 'fender' in carpart:
                            continue
                        max_score = 0
                        dam='crack'
                        overlap_area = cv2.bitwise_and(carpart_mask, pred_damage)
                        m_damage_pixel=cv2.countNonZero(overlap_area)
                        if m_damage_pixel >  damage_pixel:                            
                            if m_damage_pixel / min(cv2.countNonZero(pred_damage),cv2.countNonZero(carpart_mask)) < 0.1:
                                continue
                            if m_damage_pixel / max(cv2.countNonZero(pred_damage),cv2.countNonZero(carpart_mask)) > 0.75:
                                dam='missing'

                            damage_pixel=m_damage_pixel
                            damage_score = (dam, d_score.item(),1)
                            cat=carpart

                    
                    if damage_pixel >0:
                        checked_mask=True 
                        if cat not in final_output.keys():
                            final_output[cat] = []
                            final_output[cat].append(damage_score)
                        else:
                            final_output[cat].append(damage_score)
                            
            if not checked_mask:

                if d_score.item() < 0.56:
                    continue 

                bbox_int = np.array(bbox).astype(np.int32)
                x,y,x1,y1= bbox[:]
                crack_bbox = id_mask[y:y1,x:x1]
                (uniques, counts)=np.unique(crack_bbox, return_counts=True)
                
                if len(counts)==0 or  (len(counts)==1 and uniques[0]==0) :
                    continue 

                carpart_id = [x for x, _ in
                    sorted(zip(uniques, counts), key=lambda pair: pair[1])]

                counts = sorted(counts, key=lambda x: x)

                print(carpart_id,counts,carpart_labels)
                crack_carpart =carpart_id[-1]
                if crack_carpart==0:
                    
                    if max(counts)/sum(counts) >0.6:
                        continue 
                    crack_carpart=carpart_id[-2]
                
                cat=carpart_labels[crack_carpart-1]

                if ('door' in cat  and carpart_info['view'] is None) or  'tyre' in cat:
                    continue 
                if cat not in final_output.keys():
                    final_output[cat] = []
                final_output[cat].append(('crack',float(d_score.item()),1))

        return pred_json,final_output


    def get_damage2carpart(self,image_path,pred_json,final_output):
        pred_json,final_output=self.get_damage(image_path,pred_json,final_output)
        return self.component.get_damage2carpart(image_path,pred_json,final_output)

                                                

