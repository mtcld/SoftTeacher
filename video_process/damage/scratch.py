import cv2
import os 
import torch 
import torchvision.ops.boxes as bops

from video_process.utils.util import is_eligible_damage_on_part
from video_process.damage import DamageDecorator,FilterDamage,FilterDecorator
from video_process import yolo_models
# from car_damage.config.cfg import envs
# from video_process.utils.visualize import visual_yolo_result


class RemoveScratchByYolo(FilterDecorator):    
    def __call__(self,image,pred_json):
        
        boxes,confs,_=yolo_models['scratch'](image)
        # if envs.WHICH=='DEV-LOCAL':
        #     visual_yolo_result(img_path,boxes,confs,'scratch')
        for i in range(len(pred_json["scratch"]["bboxes"])-1,-1,-1):
            # if pred_json["scratch"]["scores"][i] >0.935:
            #     continue 

            overlap=False
            for rec in boxes:
                box1 = torch.tensor([pred_json["scratch"]["bboxes"][i]], dtype=torch.float)
                box2 = torch.tensor([rec], dtype=torch.float)
                iou = bops.box_iou(box1, box2)
                if iou[0][0] > 0.01:

                    overlap=True
                    break
            if not overlap:
                pred_json['scratch']['labels'].pop(i)
                pred_json["scratch"]["masks"].pop(i)
                pred_json["scratch"]["bboxes"].pop(i)
                pred_json["scratch"]["scores"].pop(i)

        return self.component(image,pred_json)


filter=FilterDamage()
filter=RemoveScratchByYolo(filter)


class Scratch(DamageDecorator):
    def get_damage2carpart(self,image,pred_json,final_output):
        carpart_info= pred_json['carpart']
        carpart_labels= carpart_info['labels']
        # print('len(carpart_labels)',len(carpart_labels))
        if ('grille' not in carpart_labels  and len(carpart_labels) >13)or carpart_info['view'] is None:
            return self.component.get_damage2carpart(image,pred_json,final_output)

        pred_json=filter(image,pred_json)

        edge_mask =carpart_info['edge_mask']
        for id_carpart,(cat,carpart_mask) in enumerate(zip(carpart_info['labels'], carpart_info["masks"])):
            max_score = 0
            dam='scratch'
            if not is_eligible_damage_on_part(cat, dam):
                continue
            damage_res, pred_damage_list,score_list,bboxes_list=pred_json[dam]["labels"], pred_json[dam]["masks"],pred_json[dam]["scores"],pred_json[dam]["bboxes"]
            

            for kn in range(len(pred_damage_list)):
                ## TODO : add filter related to lower confident score here
                #
                # 

                pred_damage = pred_damage_list[kn]
                if cv2.countNonZero(pred_damage)==0:
                    pred_damage=create_pseudo_mask(pred_damage,bboxes_list[kn])

                overlap_area = cv2.bitwise_and(edge_mask, pred_damage)
                damage_pixel = cv2.countNonZero(overlap_area)

                if min(cv2.countNonZero(pred_damage),cv2.countNonZero(edge_mask)) == 0:
                    continue
                if damage_pixel / min(cv2.countNonZero(pred_damage),cv2.countNonZero(edge_mask)) > 0.8:
                    continue
                overlap_area = cv2.bitwise_and(carpart_mask, pred_damage)
                damage_pixel=cv2.countNonZero(overlap_area)

                if damage_pixel > 0:
                    if damage_pixel / min(cv2.countNonZero(pred_damage),cv2.countNonZero(carpart_mask)) < 0.05:
                        continue
                    # if damage_pixel / max(cv2.countNonZero(pred_damage),cv2.countNonZero(carpart_mask)) > 0.75:
                    #     dam='missing

                    damage_score = (dam, score_list[kn],kn,id_carpart)
                    if cat not in final_output.keys():
                        final_output[cat] = []
                        final_output[cat].append(damage_score)
                    else:
                        final_output[cat].append(damage_score)

        return self.component.get_damage2carpart(image,pred_json,final_output)


def create_pseudo_mask(pred_damage,bbox):
    x1,y1,x2,y2=bbox
    center_x=int((x1+x2)/2)
    center_y=int((y1+y2)/2)
    radius= max(10,int(0.1*(min(x2-x1,y2-y1))))
    pred_damage = cv2.circle(pred_damage, (center_x,center_y), radius, (1), -1)
    return pred_damage


