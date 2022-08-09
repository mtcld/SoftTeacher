import cv2
import numpy as np
from car_damage.postprocess.utils.mask import create_id_mask
from car_damage.postprocess.utils.mask import expend_mask
from car_damage.postprocess.damage import DamageDecorator


def reject_loose(loose_damage,carpart_contours):
    id_mask=create_id_mask(carpart_contours,loose_damage.shape[:2])
    im2, contours, hierarchy = cv2.findContours(loose_damage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h =0,0,1,1
    for cont in contours:
        x1,y1,w1,h1= cv2.boundingRect(cont)
        if w1*h1 > w*h:
           x,y,w,h = x1,y1,w1,h1
    kernel =id_mask[y:y+h,x:x+w]
    # reject loose if loose is on boder of car
    if np.unique(kernel).shape[0]==2 and 0 in np.unique(kernel):
        return True
    return False

class Loose(DamageDecorator):
    def get_damage2carpart(self,image_path,pred_json,final_output):
        carpart_info= pred_json['carpart']
        carpart_labels= carpart_info['labels']
        if len(carpart_labels) <5:
            return  self.component.get_damage2carpart(image_path,pred_json,final_output)

        for cat,carpart_mask in zip(carpart_info['labels'],carpart_info["masks"]):
            max_score = 0

            dam='loose'
            damage_res, pred_damage_list,score_list= pred_json[dam]["labels"], pred_json[dam]["masks"],pred_json[dam]["scores"]
            
            ndamage_per_carpart=0
            for kn in range(len(pred_damage_list)):
                if score_list[kn] <0.965 :
                    continue 
                pred_damage = pred_damage_list[kn]


                # replace loose by dent if loose on  border car 
                # if (reject_loose(pred_damage,carpart_info["contours"])):
                #     dam='dent'

                pred_damage = expend_mask(pred_damage)

                overlap_area = cv2.bitwise_and(carpart_mask, pred_damage)
                damage_pixel=cv2.countNonZero(overlap_area)

                if damage_pixel > 0:
                    if damage_pixel / min(cv2.countNonZero(pred_damage),cv2.countNonZero(carpart_mask)) < 0.1:
                        continue
                    if damage_pixel / max(cv2.countNonZero(pred_damage),cv2.countNonZero(carpart_mask)) > 0.75:
                        dam='missing'
                    ndamage_per_carpart +=1
                    max_score = max(max_score, float(score_list[kn]))

            

            damage_score = (dam, max_score,ndamage_per_carpart)

            if max_score > 0:
                if cat not in final_output.keys():
                    final_output[cat] = []
                    final_output[cat].append(damage_score)
                else:
                    final_output[cat].append(damage_score)
        return self.component.get_damage2carpart(image_path,pred_json,final_output)