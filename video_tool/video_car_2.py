# from video_process.car.view import Car_View
import enum
from typing import Tuple
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
from video_process.Hungarian_matching import Hungarian
from video_process.matching import estimate_position


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

def carpart_model_inference(model, image,score_thre=0.5):
    result = inference_detector(model,image)
    img_,pred_boxes,pred_segms,pred_labels,pred_scores = show_result_pyplot(model, image.copy(), result,score_thr=0.1)
    
    if len(pred_boxes) == 0:
        return {'labels':[],'scores':[],'masks':[],'bboxes':[]}
    
    out_pred = []
    for pred in pred_boxes:
        box = [p for b in pred for p in b]
        out_pred.append(box)
    
    index = []

    for idx,label in enumerate(pred_labels):
        if any([i in model.CLASSES[label] for i in ['door','fender','quarter_panel']]):
            if pred_scores[idx] > 0.35:
                index.append(True)
            else:
                index.append(False)
        elif any([i in model.CLASSES[label] for i in ['front_bumper','rear_bumper']]) :
            if pred_scores[idx] > 0.5:
                index.append(True)
            else:
                index.append(False)
        else:
            if pred_scores[idx] > score_thre:
                index.append(True)
            else:
                index.append(False)
    
    pred_labels = pred_labels[index]
    out_pred = np.array(out_pred)[index].tolist()
    pred_scores = pred_scores[index]
    pred_segms = pred_segms[index]

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
    out['carpart'] = carpart_model_inference(models['carpart'],image,0.5)
    
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
    # if start_angle > 180 : 
    #     start_angle = start_angle - 360
    
    # if end_angle > 180:
    #     end_angle = end_angle - 360
    
    # if start_angle < -90 : 
    #     start_angle = start_angle + 360
    
    # if end_angle < -90:
    #     end_angle = end_angle + 360

    if abs(end_angle - start_angle) > 300:
        #end_angle = end_angle + 360
        ns = min(start_angle,end_angle)
        ne = max(start_angle,end_angle)
        ns = ns + 360

        start_angle = ns
        end_angle = ne

    # print('debug loading angle : ',start_angle,end_angle,' | ',start_angle-90,end_angle-90)
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

def collect_final_result(damaged_by_bin_json):
    confirm = {}
    for k,value in damaged_by_bin_json.items():
    #     print(value)
        for cp, damages in value.items():
            for d in damages:
                label = cp+'_'+d[0]
                if label not in confirm:
                    confirm[label] = 1
                else:
                    confirm[label] += 1
    
    return confirm

def convert_view_to_bin(view_json,bin_length):
    bin_dict = {}
    for v,info in view_json.items():
        bin = int(v) // bin_length
        if bin not in bin_dict.keys():
            bin_dict[bin] = info
        else:
            bin_dict[bin].extend(info)

    for bin,info in bin_dict.items():
        if len(info)//3 != 2*len(info)//3:
            bin_dict[bin] = [info[len(info)//3],info[2*len(info)//3]]
        else:
            bin_dict[bin] = [info[len(info)//2]]

    return bin_dict

def get_label(label):
    return label[:label.find('+')]

def main():
    
    bin_length = 5
    icon = cv2.imread('video_process/icon_car.png')
    
    icon = cv2.circle(icon.copy(),center,radius,(255,0,0),10)


    damage = Damage()
    damage = Dent(damage)
    damage = Scratch(damage)

    damage_clrs = {'scratch':np.array([255,0,0]),'dent':np.array([0,255,0]),'crack':np.array([0,0,255])} 
    track_carpart_flag  = {'rbu_rear_bumper':False,'fbu_front_bumper':False,'mirror':False}

    # video_files = glob.glob('video/video_2/*.MOV')
    # video_files=['video/IMG_5424.MOV','video/20220504_142734.mp4','video/IMG_5959.MOV','video/video_2/FBAI_Car_01.MOV','video/IMG_5942.MOV','video/IMG_5435.MOV','video/IMG_5946.MOV','video/video_2/FBAI_Car_05.MOV']
    # video_files = glob.glob('video/IMG_5946.MOV')
    video_files = ['video_tool/demo_4.avi']
    # video_files = glob.glob('video_tool/demo.avi')
    out_path = 'video_out'
    for file in video_files:
        # cap = cv2.VideoCapture('video/VID_20220709_122339.mp4')
        view_dict_by_id = {}
        view_id = 0

        name = os.path.basename(file)
        name = name[:name.rfind('.')]

        images_video_path = out_path+'/'+name+'_bin_'+str(bin_length)
        os.system('rm -rf '+images_video_path)
        Path(images_video_path).mkdir(parents=True,exist_ok=True)

        cap = cv2.VideoCapture(file)
        frame_w = int(cap.get(3))
        frame_h = int(cap.get(4))
        # video_writer = cv2.VideoWriter('video_tool/'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),4,(frame_w,frame_h))
        # video_writer = cv2.VideoWriter('video_tool/demo_4.avi',cv2.VideoWriter_fourcc('M','J','P','G'),24,(frame_w,frame_h)) 
        count = 0
        views = EMA(span=15)


        pre_v = 0
        curr_v = 0

        damages_by_frame = {}
        damaged_carpart = {}
        damaged_by_bin = {}

        hungarian = Hungarian(frame_w,frame_h)
        tracking_flag = False
        roi_list = []
        # sucess_tracking_flag = True
        record_flag = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret == True :             
                if count % 1 == 0:
                    draw_frame = frame.copy()
                    check_relabel_flag = False
                    tracking_info = []
                    if tracking_flag:
                        for roi_data in roi_list:
                            dst = estimate_position(pre_frame,frame,roi_data[1])
                            
                            if dst is None:
                                # if sucess_tracking_flag:
                                #     dst = roi_data[1]
                                # else:
                                #     sucess_tracking_flag = False
                                continue

                            roi_info = [roi_data[0],np.array(dst).reshape(-1,2).tolist(),roi_data[2]]
                            tracking_info.append(roi_info)

                            #draw tracking info
                            p1 = np.int32(dst)[2][0]
                            p2 = np.int32(dst)[0][0]
                            cv2.rectangle(draw_frame,p1,p2,(255,0,0),2)
                            #cv2.polylines(draw_frame,[np.int32(dst)],True,[255,0,0],2,cv2.LINE_AA)
                            
                            cv2.putText(draw_frame,roi_data[0],(p1[0]-130,p1[1]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(255,0,0),thickness=2)
                        
                        pre_frame = frame.copy()
                        if  count % 3 == 0:
                            tracking_flag = False
                    
                    if not tracking_flag :
                        result = get_model_prediction(frame)

                        # for b in result[0]['car']['bboxes']:
                        #     cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,255,0),5)
                        # print('debug : ',result[0]['car']['labels'])

                        result = carpart_if.add_carpart_info(['krug/'],result)

                        roi_list = []
                        tmp_track_carpart_flag  = {'rbu_rear_bumper':False,'fbu_front_bumper':False,'mirror':False}
                        for idx,b in enumerate(result[0]['carpart']['bboxes']):
                            label_carpart = get_label(result[0]['carpart']['labels'][idx])
                            if any([i in label_carpart for i in ['door','fender','quarter_panel']]):
                                # if tmp_track_carpart_flag[label_carpart] and track_carpart_flag[label_carpart]:
                                roi = [[int(b[0]),int(b[1])],[int(b[2]),int(b[1])],[int(b[2]),int(b[3])],[int(b[0]),int(b[3])]]
                                roi_list.append([result[0]['carpart']['labels'][idx],roi,idx])
                                tracking_flag = True

                                #draw detection info
                                cv2.rectangle(draw_frame,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
                                # score = round(result[0]['carpart']['scores'][idx],2)
                                cv2.putText(draw_frame,result[0]['carpart']['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,0,255),thickness=2)
                            elif any([i in label_carpart for i in ['front_bumper','rear_bumper','mirror']]):
                                tmp_track_carpart_flag[label_carpart] = True

                                if tmp_track_carpart_flag[label_carpart] and track_carpart_flag[label_carpart]:
                                    roi = [[int(b[0]),int(b[1])],[int(b[2]),int(b[1])],[int(b[2]),int(b[3])],[int(b[0]),int(b[3])]]
                                    roi_list.append([result[0]['carpart']['labels'][idx],roi,idx])
                                    tracking_flag = True

                                    #draw detection info
                                    cv2.rectangle(draw_frame,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
                                    # score = round(result[0]['carpart']['scores'][idx],2)
                                    cv2.putText(draw_frame,result[0]['carpart']['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,0,255),thickness=2)
                        
                        track_carpart_flag = tmp_track_carpart_flag.copy()
                        
                        pre_frame = frame.copy()

                        # print('debug v:',result[0]['carpart']['view'])
                        if result[0]['carpart']['view'] is None:
                            print('alo')
                            continue
                        
                        roi_list, result, check_relabel_flag,view_dict_by_id = hungarian.bipartite_matching(tracking_info,roi_list,result,view_dict_by_id,view_id,name)
                        
                        v = views.add(result[0]['carpart']['view'])
                        print('view : ',v,count)
                        v = int(v)

                        if v <5: 
                            record_flag = True
                        curr_v = v
                        if count == 0:
                            pre_v = v

                        # icon = draw_icon(icon,pre_v,curr_v)
                        # pre_v = curr_v

                        view_info = {'view': v,'frame':frame,'result':result}
                        view_dict_by_id[view_id] = view_info
                        view_id += 1

                        # if v not in view_dict.keys():
                        #     view_dict[v] = [view_info]
                        # else:
                        #     view_dict[v].append(view_info)
                    
                    # if record_flag :
                    #     video_writer.write(frame)

                    # if curr_v // bin_length not in damages_by_frame.keys() :
                        # result = damage_model_inference(dent_model,result,frame,score_thre=0.35)
                        # result = damage_model_inference(scratch_model,result,frame,score_thre=0.6)

                        # pred_result,damage_result = compare_masks(frame,damage,result[0])

                        # print('debug final output : ',damage_result)
                        # for k in damage_result.keys():
                        #     if k in damaged_carpart.keys():
                        #         damaged_carpart[k] += 1
                        #     else:
                        #         damaged_carpart[k] = 1

                        # damages_by_frame[curr_v // bin_length] = damage_result
                        # print('debug damages_by_frame : ',damages_by_frame)

                        # # draw scratch

                        # for mask in pred_result['scratch']['masks']:
                        #     mask = mask.astype(bool)
                        #     frame[mask] = frame[mask]*0.5+damage_clrs['scratch']*0.5
                        
                        # for mask in pred_result['dent']['masks']:
                        #     mask = mask.astype(bool)
                        #     frame[mask] = frame[mask]*0.5+damage_clrs['dent']*0.5

                    ## draw main car and its damaged car parts
                    # for carpart, confirmation in damaged_carpart.items():
                    #     if carpart in result[0]['carpart']['labels']:
                    #         index = result[0]['carpart']['labels'].index(carpart)
                    #         mask = result[0]['carpart']['masks'][index]
                    #         mask = mask.astype(bool)
                    #         if confirmation == 1:
                    #             clr = np.array([0,255,0])
                    #         else:
                    #             clr = np.array([0,0,255])

                    #         frame[mask] = frame[mask]*0.5 + clr*0.5

                    # main_car = result[0]['car']['main_car']

                    # cv2.rectangle(frame,(main_car[0],main_car[1]),(main_car[0]+ main_car[2],main_car[1]+main_car[3]),(255,0,0),5)

                    # frame = cv2.putText(frame,str(result[0]['carpart']['view'])+ ' | '+str(v),(main_car[0],main_car[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                    # cv2.imwrite('debug_icon.jpg',icon)
                    cv2.imwrite('video_tool/debug_view.jpg',draw_frame)

                    
                    # if check_relabel_flag : 
                    #     cv2.putText(draw_frame,'relabel',(30,30),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,0,255),thickness=2)
                    #     video_writer.write(draw_frame)
                    # if check_relabel_flag:
                    #     video_writer.write(frame)
                    # if count < 120:
                    #     video_writer.write(frame)
                    
                    # video_writer.write(draw_frame)
                
                count += 1
            else:
                break
        view_dict = {}

        for view_id, view_info in view_dict_by_id.items():
            v = view_info['view']
            if v not in view_dict.keys():
                view_dict[v] = [view_info]
            else:
                view_dict[v].append(view_info)

        del view_dict_by_id
        
        bin_dict = convert_view_to_bin(view_dict,bin_length)
        del view_dict
        print('bin_length : ',bin_length)

        Path(images_video_path+'/output').mkdir(parents=True,exist_ok=True)
        ### run damages inference 
        for bin, infos in bin_dict.items():
            for idx,info in enumerate(infos) : 
                print(bin, ' : ',info['view'])
                result = info['result']
                image_out_name = 'bin_'+str(bin)+'_'+str(idx)+'_frame_'+str(info['view'])+'.jpg'
                cv2.imwrite(images_video_path+'/'+image_out_name,info['frame'])
                result = damage_model_inference(dent_model,result,info['frame'],score_thre=0.35)
                result = damage_model_inference(scratch_model,result,info['frame'],score_thre=0.6)

                pred_result,damage_result = compare_masks(info['frame'],damage,result[0])

                # draw output
                # draw carpart 
                output_image = info['frame'].copy()
                for idb,box in enumerate(result[0]['carpart']['bboxes']):
                    # print('some scratch')
                    # if 'door' in result[0]['carpart']['labels'][idb]:
                    #     mask = result[0]['carpart']['masks'][idb].astype(bool)
                    #     output_image[mask] = output_image[mask]*0.5+np.array([255,0,0])*0.5
                        
                    cv2.rectangle(output_image,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
                    cv2.putText(output_image,result[0]['carpart']['labels'][idb],(box[0],box[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(255,0,0),thickness=2)
                
                for idm,mask in enumerate(pred_result['scratch']['masks']):
                    # print('some scratch')
                    mask = mask.astype(bool)
                    output_image[mask] = output_image[mask]*0.5+damage_clrs['scratch']*0.5
                    # b = pred_result['scratch']['bboxes'][idm]
                    # print(b,damage_clrs['scratch'])
                    # cv2.rectangle(output_image,(b[0],b[1]),(b[2],b[3]),(255,0,0),2)
                
                for mask in pred_result['dent']['masks']:
                    # print('some dent')
                    mask = mask.astype(bool)
                    output_image[mask] = output_image[mask]*0.5+damage_clrs['dent']*0.5
                
                cv2.imwrite(images_video_path+'/output/'+image_out_name,output_image)

                
                damaged_by_bin[str(bin)+'_'+str(idx)] = damage_result

            # print('view : ',info['view'])
        # cleaned_damaged_by_bin = clean_outlier(damaged_by_bin,bin_length)
        cleaned_damaged_by_bin = damaged_by_bin
        final_result = collect_final_result(damaged_by_bin)
        print(final_result)

        with open(out_path+'/'+name+'_damges_by_frame_v2.json', 'w', encoding='utf-8') as f:
            json.dump({'damaged_bin':cleaned_damaged_by_bin,'result':final_result}, f, ensure_ascii=False, indent=4)
        # with open(out_path+'/'+name+'_damges_by_frame.json', 'w', encoding='utf-8') as f:
        #     json.dump({'origin':damaged_by_bin,'cleaned':cleaned_damaged_by_bin}, f, ensure_ascii=False, indent=4)



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