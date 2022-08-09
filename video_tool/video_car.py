# from video_process.car.view import Car_View
from video_process.carpart.segment import CarpartInfo
from video_process.damage import Damage
from video_process.damage.scratch import Scratch
from video_process.damage.dent import Dent

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

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector as st_init_detector


cp_model = init_detector('checkpoint/carpart_rear_exp_2/carpart_rear.py',
                      'checkpoint/carpart_rear_exp_2/epoch_29.pth',device='cuda:0')

car_model = init_detector('thirdparty/mmdetection/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
                          'checkpoint/coco_pretrain/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth',device='cuda:0')


scratch_model = init_detector('checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/swa-scratch-copy-paste-HSV-LSJ.py',
                        'checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/best_bbox_mAP.pth',device='cuda:0')

dent_model = st_init_detector('checkpoint/19_02_2022/dent/dent_mask.py',
                        'checkpoint/19_02_2022/dent/iter_207000.pth',device='cuda:0')

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

def main():
    
    bin_length = 5

    damage = Damage()
    damage = Dent(damage)
    damage = Scratch(damage)

    damage_clrs = {'scratch':np.array([255,0,0]),'dent':np.array([0,255,0]),'crack':np.array([0,0,255])} 

    # video_files = glob.glob('video/*.MOV')
    # video_files = glob.glob('video/IMG_5427.MOV')
    video_files = ['video/video_2/FBAI_Car_05.MOV','video/20220504_142734.mp4']
    out_path = 'video_out'
    for file in video_files:
        # cap = cv2.VideoCapture('video/VID_20220709_122339.mp4')
        icon = cv2.imread('video_process/icon_car.png')
        icon = cv2.circle(icon.copy(),center,radius,(255,0,0),10)
        
        name = os.path.basename(file)
        name = name[:name.rfind('.')]

        images_video_path = out_path+'/'+name+'_bin_'+str(bin_length)+'_first_frame'
        Path(images_video_path).mkdir(parents=True,exist_ok=True)

        cap = cv2.VideoCapture(file)
        frame_w = int(cap.get(3))
        frame_h = int(cap.get(4))
        video_writer = cv2.VideoWriter(out_path+'/'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),24,(frame_w*2,frame_h*2)) 
        count = 0
        views = EMA(span=15)

        pre_v = 0
        curr_v = 0

        damages_by_frame = {}
        damaged_carpart = {}

        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret == True :             
                if count % 1 == 0:
                    result = get_model_prediction(frame)

                    # for b in result[0]['car']['bboxes']:
                    #     cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,255,0),5)
                    # print('debug : ',result[0]['car']['labels'])

                    result = carpart_if.add_carpart_info(['krug/'],result)

                    
                    # print('debug v:',v)
                    if result[0]['carpart']['view'] is None:
                        # print('alo')
                        continue

                    v = views.add(result[0]['carpart']['view'])
                    v = int(v)
                    curr_v = v
                    if count == 0:
                        pre_v = v
                    
                    icon = draw_icon(icon,pre_v,curr_v)
                    icon_frame = cv2.resize(icon.copy(),(frame_w,frame_h))
                    pre_v = curr_v
                    if curr_v // bin_length not in damages_by_frame.keys() :
                        cv2.imwrite(images_video_path+'/'+'bin_'+str(curr_v // bin_length)+'_frame_'+str(curr_v)+'.jpg',frame)

                        result = damage_model_inference(dent_model,result,frame,score_thre=0.35)
                        result = damage_model_inference(scratch_model,result,frame,score_thre=0.6)

                        pred_result,damage_result = compare_masks(frame,damage,result[0])

                        print('debug final output : ',damage_result)
                        for k in damage_result.keys():
                            if k in damaged_carpart.keys():
                                damaged_carpart[k] += 1
                            else:
                                damaged_carpart[k] = 1

                        damages_by_frame[curr_v // bin_length] = damage_result
                        print('debug damages_by_frame : ',damages_by_frame)

                        # draw scratch
                        damages_frame = frame.copy()
                        for mask in pred_result['scratch']['masks']:
                            mask = mask.astype(bool)
                            damages_frame[mask] = damages_frame[mask]*0.5+damage_clrs['scratch']*0.5
                        
                        for mask in pred_result['dent']['masks']:
                            mask = mask.astype(bool)
                            damages_frame[mask] = damages_frame[mask]*0.5+damage_clrs['dent']*0.5

                    ## draw main car and its damaged car parts
                    carpart_frame = frame.copy()
                    for carpart, confirmation in damaged_carpart.items():
                        if carpart in result[0]['carpart']['labels']:
                            index = result[0]['carpart']['labels'].index(carpart)
                            mask = result[0]['carpart']['masks'][index]
                            mask = mask.astype(bool)
                            if confirmation == 1:
                                clr = np.array([0,255,0])
                            else:
                                clr = np.array([0,0,255])

                            carpart_frame[mask] = carpart_frame[mask]*0.5 + clr*0.5

                    main_car = result[0]['car']['main_car']

                    cv2.rectangle(carpart_frame,(main_car[0],main_car[1]),(main_car[0]+ main_car[2],main_car[1]+main_car[3]),(255,0,0),5)

                    # frame = cv2.putText(frame,str(result[0]['carpart']['view'])+ ' | '+str(v),(main_car[0],main_car[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                    # cv2.imwrite('debug_icon.jpg',icon)
                    
                    output_frame = np.hstack((np.vstack((frame,carpart_frame)),np.vstack((damages_frame,icon_frame))))
                    cv2.imwrite('debug_view.jpg',output_frame)
                    video_writer.write(output_frame)
                
                count += 1
            else:
                break

        with open(out_path+'/'+name+'_damges_by_first_frame_in_bin.json', 'w', encoding='utf-8') as f:
            json.dump(damages_by_frame, f, ensure_ascii=False, indent=4)



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