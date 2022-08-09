import cv2
import random
from car_damage.postprocess.car.view.three_d_car import color_segment
from car_damage.postprocess.damage import DamageDecorator
from car_damage.postprocess.damage.missing.adjust_position import missing_relation,get_position_delta,adjust_position_mask
from car_damage.postprocess.damage.missing.adjust_scale import adjust_scale_mask,scale_mask
from car_damage.postprocess.utils.mask import process_front_bumper_mask

import cv2
import numpy as np
import torch 
import os 
import torchvision.ops.boxes as bops
from car_damage.postprocess.models import yolo_models
from car_damage.config.cfg import envs
from car_damage.postprocess.utils.visualize import visual_yolo_result

class RemoveMissingByYolo():    
    def __call__(self,img_path,pred_json):
        boxes,confs,_=yolo_models['missing'](img_path)
        if envs.WHICH=='DEV-LOCAL':
            visual_yolo_result(img_path,boxes,confs,'missing')
            
        for i in range(len(pred_json["missing"]["bboxes"])-1,-1,-1):
            
            # if 0.85< pred_json["missing"]["scores"][i] <0.9:
            #     continue
            overlap=False
            for yolo_c,rec in zip(confs,boxes):
                box1 = torch.tensor([pred_json["missing"]["bboxes"][i]], dtype=torch.float)
                box2 = torch.tensor([rec], dtype=torch.float)
                iou = bops.box_iou(box1, box2)
                if iou[0][0] > 0.25 :
                    overlap=True
                    break
            if not overlap:
                pred_json["missing"]["masks"].pop(i)
                pred_json["missing"]["bboxes"].pop(i)
                pred_json["missing"]["scores"].pop(i)
        

        return pred_json

m_filter= RemoveMissingByYolo()

def check_missing_in_frontbumber(missing_relations):
    if len(missing_relations) > 1:
        return False
    if 'fbu_front_bumper+f' in missing_relations and missing_relations['fbu_front_bumper+f'].count(None)==2:
        return True  

    return False

def check_missing_in_foglight_benzel(missing_relations):
    if  'fbe_fog_light_bezel+lf' in missing_relations and missing_relations['fbe_fog_light_bezel+lf'].count(None)==2:
        return True  
    if  'fbe_fog_light_bezel+rf' in missing_relations and missing_relations['fbe_fog_light_bezel+rf'].count(None)==2:
        return True  
    return False



def is_eligible_damage_on_part(carpart, view):
   
    if 0 <=int(view)<20   and 'fender' in carpart:
        return True 
    if  180< int(view) < 330 and (('fender+rf') in carpart or 'hli_head_light+rf' in carpart):
        return True 
    if  30 <=int(view) < 90 and (('fender+lf') in carpart or 'hli_head_light+lf' in carpart):
        return True 

    return False 

def recheck_font_bumber(info_missing):
    front_bumber_area= info_missing['fbu_front_bumper+f']

    other_area = info_missing['other_carpart'] 

    print('mmmmmmmmmmmmmmmmmmmm',front_bumber_area,other_area)
    if front_bumber_area > (1 - other_area) *0.4 :
        return True
    return False


class Missing(DamageDecorator):
    def adjust_missing_masks(self,damage,carpart_infos,visual=False):
        self.proj_masks, self.proj_labels, pr_img= color_segment.get_all_projec_mask_labels(carpart_infos['totaled_info']['view_img'], True)
        x,y,w, h = carpart_infos['totaled_info']['car_bbox']
        self.proj_masks=[cv2.resize(mask, (w, h)) for mask in self.proj_masks]
        real_masks= [cv2.resize(mask, (w, h)) for mask in carpart_infos['masks']]
        
        # adjust position 
        self.missing_relations = missing_relation(carpart_infos,damage)
        print('self.missing_relations ***************',self.missing_relations)
        # self.proposed_carparts_by_relation=add_missing_from_relation(final_output, self.missing_relations)    
        delta=get_position_delta(self.proj_masks, self.proj_labels, self.missing_relations,carpart_infos)
        damage, new_contours,old_contours=adjust_position_mask(damage,[x,y,w, h],delta)

        pr_img=cv2.resize(pr_img,(w,h))


        # scale position 

        if '_c' not in carpart_infos['totaled_info']['view_img']:
            if len(carpart_infos['labels']) <15  :
                ratio_scale,miss_center,self.proj_masks = adjust_scale_mask(self.proj_masks, self.proj_labels,real_masks,carpart_infos['labels'],damage)
                pr_img=scale_mask(pr_img,miss_center,ratio_scale)

        else :
            if not check_missing_in_frontbumber(self.missing_relations):
                ratio_scale,miss_center,self.proj_masks = adjust_scale_mask(self.proj_masks, self.proj_labels,real_masks,carpart_infos['labels'],damage)
                pr_img=scale_mask(pr_img,miss_center,ratio_scale)

        # visual 
        self.proj_car_area=cv2.countNonZero(color_segment.check_carpart_in_image(pr_img, (0, 0, 0.24), "car")[0])
        if visual:
            for new_cont,old_cont in zip(new_contours,old_contours):
                # cv2.drawContours(pr_img, old_cont, -1, (0, 0, 0), 3)
                cv2.drawContours(pr_img, new_cont, -1, (255, 0, 0), 3)
            cv2.imwrite('totaled.jpg', pr_img)
        return damage


    
    def get_damage2carpart(self,image_path,pred_json,final_output):
        pred_json=m_filter(image_path,pred_json)
        dam='missing'
        missing_infos=pred_json[dam]
        carpart_infos = pred_json['carpart']

        if len(missing_infos["scores"])==0 or carpart_infos['totaled_info']['view_img']=='' or carpart_infos['view'] is None:
            return self.component.get_damage2carpart(image_path,pred_json,final_output)

        for i,damage in enumerate(missing_infos['masks']):
            damage_mask=self.adjust_missing_masks(damage,carpart_infos,False)

            
            if check_missing_in_foglight_benzel(self.missing_relations):
                if 'fbe_fog_light_bezel+lf' in self.missing_relations:
                    proj_label='fbe_fog_light_bezel+lf'
                else:
                    proj_label='fbe_fog_light_bezel+rf'

                proj_label =proj_label.replace('fbe_fog_light_bezel','fli_fog_light')
                if proj_label not in final_output:
                    final_output[proj_label] = []
                    final_output[proj_label].append(('missing', random.uniform(0.91, 0.95),1))
                    continue

            if 'fender+rf' in self.proj_labels and 'hli_head_light+rf' in self.proj_labels :
                f_i = self.proj_labels.index('fender+rf')
                hl_i= self.proj_labels.index('hli_head_light+rf')

                iv_hl= cv2.bitwise_not(self.proj_masks[hl_i])
                self.proj_masks[f_i] = cv2.bitwise_and(self.proj_masks[f_i],iv_hl)


            if 'fender+lf' in self.proj_labels and 'hli_head_light+lf' in self.proj_labels :
                f_i = self.proj_labels.index('fender+lf')
                hl_i= self.proj_labels.index('hli_head_light+lf')

                iv_hl= cv2.bitwise_not(self.proj_masks[hl_i])
                self.proj_masks[f_i] = cv2.bitwise_and(self.proj_masks[f_i],iv_hl)


            #  check hood is open
            open_hood=False
            for proj_mask,proj_label in zip(self.proj_masks,self.proj_labels):
                if 'hood' not in proj_label:
                    continue
                if is_eligible_damage_on_part(proj_label,carpart_infos['view']):
                    continue
                overlap_area =  cv2.countNonZero(cv2.bitwise_and(proj_mask, damage_mask))
                if cv2.countNonZero(proj_mask)==0 or cv2.countNonZero(damage_mask)==0 : 
                    continue
                ratio_proj_mask=overlap_area/cv2.countNonZero(proj_mask)
                ratio_damage= overlap_area/cv2.countNonZero(damage_mask)                
                if  (ratio_damage >0.8  and ratio_proj_mask >0.1) or (ratio_proj_mask>0.4 and ratio_damage>0.1):
                    if 'hood+f' in proj_label and 'hood+f' in carpart_infos['totaled_info']['carparts']:
                        open_hood=True 
            if open_hood :
                break

            #  check foglight and grille 
            missing_parts={}
            for proj_mask,proj_label in zip(self.proj_masks,self.proj_labels):
                if 'grille' not in proj_label and 'fog_light' not in proj_label and 'fbu_front_bumper' not in proj_label:
                    continue
                if is_eligible_damage_on_part(proj_label,carpart_infos['view']):
                    continue
                overlap_area =  cv2.countNonZero(cv2.bitwise_and(proj_mask, damage_mask))
                if cv2.countNonZero(proj_mask)==0 or cv2.countNonZero(damage_mask)==0 : 
                    continue
                ratio_proj_mask=overlap_area/cv2.countNonZero(proj_mask)
                ratio_damage= overlap_area/cv2.countNonZero(damage_mask)  
                             
                                
                if  (ratio_damage >0.8  and ratio_proj_mask >0.1) or (ratio_proj_mask>0.4 and ratio_damage>0.1) \
                    or ('fog_light' in proj_label and ratio_proj_mask > 0.249 and ratio_damage >0.05) or\
                     ('grille' in proj_label and ratio_proj_mask == 1.0 and ratio_damage >0.05):
                    
                    if 'grille' in proj_label and ratio_proj_mask<0.8:
                        missing_parts['grille'] = ratio_damage
             
                    if 'fog_light' in proj_label:
                        missing_parts['fog_light'] = ratio_damage
            
            front_bumper_flag=False
            info_missing={'other_carpart':0}
            for proj_mask,proj_label in zip(self.proj_masks,self.proj_labels):
                if 'grille' in missing_parts and 'fog_light' in missing_parts :
                    if missing_parts['grille'] < missing_parts['fog_light']:
                        if 'grille' in proj_label:
                            continue
                    else :
                        if 'fog_light' in proj_label:
                            continue
                if is_eligible_damage_on_part(proj_label,carpart_infos['view']):
                    continue

                damage_mask_cp=damage_mask.copy()    
                if 'front_bumper' in proj_label :
                    proj_mask,r_view,center_bumper_x=process_front_bumper_mask(self.proj_labels,self.proj_masks,proj_mask)
                    or_h,or_w = damage_mask_cp.shape[:2]
                    if r_view :
                        damage_mask_cp=damage_mask_cp[0:or_h,0:center_bumper_x]
                        damage_mask_cp = cv2.copyMakeBorder(damage_mask_cp, 0, 0, 0, or_w-center_bumper_x, cv2.BORDER_CONSTANT)
                    else:
                        damage_mask_cp=damage_mask_cp[0:or_h,center_bumper_x:or_w-1]
                        damage_mask_cp = cv2.copyMakeBorder(damage_mask_cp, 0, 0, center_bumper_x+1, 0, cv2.BORDER_CONSTANT)

                    for label1, mask1 in zip(self.proj_labels,self.proj_masks):
                        if 'grille' in label1 or 'fog_light' in label1 or 'flp_front_license_plate' in label1:
                            mask1=cv2.bitwise_not(mask1)
                            proj_mask=cv2.bitwise_and(proj_mask, mask1) 
             
                overlap_area =  cv2.countNonZero(cv2.bitwise_and(proj_mask, damage_mask_cp))

                # overlap_area = overlap_area*pred_json['carpart']['car_area']/self.proj_car_area

                if cv2.countNonZero(proj_mask)==0 or cv2.countNonZero(damage_mask_cp)==0 : 
                    continue
                
                ratio_proj_mask=overlap_area/cv2.countNonZero(proj_mask)
                ratio_damage= overlap_area/cv2.countNonZero(damage_mask_cp)
                print(proj_label,'ratio_proj_mask',ratio_proj_mask,'ratio_damage',ratio_damage)

                if 'fbu_front_bumper+f' not in proj_label and ratio_damage>0.1:
                    info_missing['other_carpart'] +=ratio_damage
                if  (ratio_damage >0.8  and ratio_proj_mask >0.1) or (ratio_proj_mask>0.4 and ratio_damage>0.1) \
                    or ('fog_light' in proj_label and ratio_proj_mask > 0.249 and ratio_damage >0.05) or\
                     ('mirror' in proj_label and ratio_proj_mask > 0.98 and ratio_damage >0.05) or \
                     ('grille' in proj_label and ratio_proj_mask == 1.0 and ratio_damage >0.05) or \
                     ('hli_head_light' in proj_label and ratio_proj_mask == 1.0 and ratio_damage >0.05):

                    if 'fbu_front_bumper+f' in proj_label:
                        if ratio_proj_mask > 0.5 :
                            front_bumper_flag=True
                            info_missing[proj_label] = ratio_damage
                        else:
                            continue
                    
                    if 'hood+f' in proj_label and 'hood+f' in carpart_infos['totaled_info']['carparts']:
                        continue 
                    # if 'fbu_front_bumper+f' in proj_label and 'fbu_front_bumper+f' in carpart_infos['totaled_info']['carparts']:
                    #     continue 

                    if 'rbu_rear_bumper+b' in proj_label and 'rbu_rear_bumper+b' in carpart_infos['totaled_info']['carparts']:
                        continue 
                    if 'tail_gate+b' in proj_label and 'tail_gate+b' in carpart_infos['totaled_info']['carparts']:
                        continue
                    if 'fog_light' in proj_label and proj_label in carpart_infos['totaled_info']['carparts']: 
                        continue 

                    if 'door' in proj_label and proj_label in carpart_infos['totaled_info']['carparts']:
                        proj_label=proj_label.replace('door','window')

                    # if 'fog_light' in proj_label and 'fli_fog_light+lf' not in  carpart_infos['totaled_info']['carparts'] and   'fli_fog_light+rf' not in carpart_infos['totaled_info']['carparts']:
                    #     proj_label='grille+f'
                    
                    if proj_label not in final_output:
                        final_output[proj_label] = []

                    final_output[proj_label].append(('missing', random.uniform(0.91, 0.95),1))
                    continue
            
            if front_bumper_flag and recheck_font_bumber(info_missing):
                if recheck_font_bumber(info_missing) ==False:
                    continue 
                proj_label='fbu_front_bumper+f'
                if proj_label not in final_output:
                    final_output[proj_label] = []
                final_output[proj_label].append(('missing', random.uniform(0.91, 0.95),1))
                
                    

        return self.component.get_damage2carpart(image_path,pred_json,final_output)

