from curses import flash
import enum
from unittest import result
from video_process import yolo_models
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector as st_init_detector

import numpy as np
import cv2
import os
import glob
# from video_process import Hungarian_matching
import json
from pathlib import Path

from video_process.Hungarian_matching import Hungarian
from video_process.matching import estimate_position
from video_process.carpart.segment import CarpartInfo
from video_process.EMA import EMA
from video_process.video_utils import compare_masks, filter_carpart_by_view, collect_final_result_after_cross_check
from video_process.Cross_checking import Cross_check_pair
from video_process.damage import Damage
from video_process.damage.scratch import Scratch
from video_process.damage.dent import Dent
from video_process.carpart.filter_carpart import correct_quarter_panel_base_fuel_tank_door

damage_filter = Damage()
damage_filter = Dent(damage_filter)
damage_filter = Scratch(damage_filter)

# from video_process.damage import scratch

cp_model = init_detector('checkpoint/carpart_rear_exp_2/carpart_rear.py',
                      'checkpoint/carpart_rear_exp_2/epoch_29.pth',device='cuda:0')

car_model = init_detector('thirdparty/mmdetection/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
                          'checkpoint/coco_pretrain/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth',device='cuda:0')

models = {'car':car_model,'carpart':cp_model}

carpart_filter = CarpartInfo()

def get_car_carpart_model_prediction(pred_json,image):
    pred_json[0]['car'] = car_model_inference(models['car'],image)
    pred_json[0]['carpart'] = carpart_model_inference(models['carpart'],image,0.5)
    
    return pred_json

def get_car_carpart_model_prediction(image):
    out = {}
    out['car'] = car_model_inference(models['car'],image)
    out['carpart'] = carpart_model_inference(models['carpart'],image,0.5)
    
    return out

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

    return {"labels": [model.CLASSES[l] for l in pred_labels], "scores": pred_scores.tolist(), "masks": [s.astype(np.uint8) for s in pred_segms],'bboxes':out_pred}

def create_pseudo_mask(image,bbox):
    mask = np.zeros_like(image[:,:,0]).astype(np.uint8)

    x1,y1,x2,y2=bbox
    center_x=int((x1+x2)/2)
    center_y=int((y1+y2)/2)
    radius= max(10,int(0.1*(min(x2-x1,y2-y1))))
    
    mask = cv2.circle(mask, (center_x,center_y), radius, (1), -1)
    
    return mask

def yolo_damage_inference(image):
    pred_json = {}
    flag = False
    
    scratch_boxes, scratch_confs,_ = yolo_models['scratch'](image)
    scratch_masks = [create_pseudo_mask(image,box) for box in scratch_boxes]
    scratch_labels = ['scratch' for b in scratch_boxes]

    pred_json['scratch'] = {'labels':scratch_labels,'scores':scratch_confs,'masks':scratch_masks,'bboxes':scratch_boxes}

    dent_boxes, dent_confs,_ = yolo_models['dent'](image)
    dent_masks = [create_pseudo_mask(image,box) for box in dent_boxes]
    dent_labels = ['dent' for b in dent_boxes]

    pred_json['dent'] = {'labels':dent_labels,'scores':dent_confs,'masks':dent_masks,'bboxes':dent_boxes}

    if len(scratch_boxes) > 0 or len(dent_boxes) > 0:
        flag = True

    return [pred_json],flag

class FrameQueue():
    def __init__(self):
        self.queue = []
        self.frame_id_list = []
        self.length = 0
        self.max_length = 10
    
    def enqueue(self,frame_info):
        if frame_info['frame_id'] in self.frame_id_list:
            return

        if self.length < self.max_length :
            self.queue.append(frame_info)
            self.frame_id_list = [i['frame_id'] for i in self.queue]
            self.length += 1

            return
        
        self.queue.pop(0)
        self.queue.append(frame_info)
        self.frame_id_list = [i['frame_id'] for i in self.queue]
    
    def get_queue(self):
        return self.queue

def draw_result(pred_json,image):
    pred_json = pred_json[0]

    for cate in pred_json.keys():
        # print(cate)
        if cate not in ['car','carpart']:
            for idx,box in enumerate(pred_json[cate]['bboxes']):
                image = cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
                cv2.putText(image,pred_json[cate]['labels'][idx],(box[0],box[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,255,0),thickness=2)

        if cate == 'carpart':
            for idx,b in enumerate(pred_json[cate]['bboxes']):
                label_carpart = get_label(pred_json[cate]['labels'][idx])
                if any([i in label_carpart for i in ['door','fender','quarter_panel','tail_light','head_light','front_bumper','rear_bumper','mirror']]):
                    #draw detection info
                    cv2.rectangle(image,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
                    # score = round(pred_json[cate]['scores'][idx],2)
                    cv2.putText(image,pred_json[cate]['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,0,255),thickness=2)

    return image 

def get_label(label):
    return label[:label.find('+')]

def get_car_info(pred_json,car_info_dict_by_frame,frame_queue,hungarian,file_name):
    hungarian.reset_count_frame() 
    # hungarian.reset_cache()

    tracking_flag = False
    roi_list = []
    view_dict_by_id = {}
    view_id = 0

    track_label_list = ['door','fender','quarter_panel','tail_light','head_light','front_bumper','rear_bumper','mirror']

    for item in frame_queue.get_queue():
        frame = item['frame']
        frame_id = item['frame_id']

        tracking_info = []
        if tracking_flag : 
            for roi_data in roi_list:
                dst = estimate_position(pre_frame,frame,roi_data[1])

                if dst is None:
                    continue
                
                roi_estimate_info = [roi_data[0],np.array(dst).reshape(-1,2).tolist(),roi_data[2]]
                tracking_info.append(roi_estimate_info)
            
            tracking_flag = False
        
        if not tracking_flag : 
            if frame_id in car_info_dict_by_frame.keys():
                car_info = car_info_dict_by_frame[frame_id]['result']
            else: 
                print('do inference car !!')
                car_info = get_car_carpart_model_prediction(frame.copy())
                car_info = carpart_filter.add_carpart_info(['/krug'],[car_info])
                # car_info_dict[frame_id] = car_info
            
            roi_list = []

            for idx,b in enumerate(car_info[0]['carpart']['bboxes']):
                label_carpart = get_label(car_info[0]['carpart']['labels'][idx])
                if any([i in label_carpart for i in track_label_list]):
                    roi = [[int(b[0]),int(b[1])],[int(b[2]),int(b[1])],[int(b[2]),int(b[3])],[int(b[0]),int(b[3])]]
                    roi_list.append([car_info[0]['carpart']['labels'][idx],roi,idx])
                    tracking_flag = True

            roi_list, car_info, _,view_dict_by_id = hungarian.bipartite_matching_v2(tracking_info,roi_list,car_info,view_dict_by_id,view_id,frame_id,file_name)
            # car_info_dict[frame_id] = car_info

            view_info = {'frame_id':frame_id,'frame':frame,'result':car_info}
            view_dict_by_id[view_id] = view_info
            view_id += 1
        # roi_list = []
        

        # for idx, box in enumerate(car_info['carpart']):

        pre_frame = frame.copy()


    #update car_info_dict_by_frame
    for view_id, info in view_dict_by_id.items():
        # if info['frame_id'] not in car_info_dict_by_frame.keys():
        car_info_dict_by_frame[info['frame_id']] = {'result':info['result'],'frame':info['frame']}

    pred_json[0]['car'] = car_info[0]['car']
    pred_json[0]['carpart'] = car_info[0]['carpart']
    
    return pred_json

def verify_car_info_dict(car_info_dict_by_frame,hungarian):
    hungarian.reset_cache()
    car_info_dict_by_frame = dict(sorted(car_info_dict_by_frame.items(),reverse=True))
    tracking_flag = False
    roi_list = []

    view_dict_by_id = {}
    view_id = 0

    track_label_list = ['door','fender','quarter_panel','tail_light','head_light','front_bumper','rear_bumper','mirror']

    for frame_id, frame_info in car_info_dict_by_frame.items():
        frame = frame_info['frame']
        car_info = frame_info['result']

        tracking_info = []
        if tracking_flag : 
            for roi_data in roi_list:
                dst = estimate_position(pre_frame,frame,roi_data[1])

                if dst is None:
                    continue
                
                roi_estimate_info = [roi_data[0],np.array(dst).reshape(-1,2).tolist(),roi_data[2]]
                tracking_info.append(roi_estimate_info)
            
            tracking_flag = False
        
        if not tracking_flag:
            roi_list = []

            for idx,b in enumerate(car_info[0]['carpart']['bboxes']):
                label_carpart = get_label(car_info[0]['carpart']['labels'][idx])
                if any([i in label_carpart for i in track_label_list]):
                    roi = [[int(b[0]),int(b[1])],[int(b[2]),int(b[1])],[int(b[2]),int(b[3])],[int(b[0]),int(b[3])]]
                    roi_list.append([car_info[0]['carpart']['labels'][idx],roi,idx])
                    tracking_flag = True

            roi_list, car_info, _ ,view_dict_by_id= hungarian.bipartite_matching(tracking_info,roi_list,car_info,view_dict_by_id,view_id,'verify')
            
            view_info = {'frame_id':frame_id,'frame':frame,'result':car_info}
            view_dict_by_id[view_id] = view_info
            view_id += 1
            
            car_info_dict_by_frame[frame_id]['result'] = car_info

        pre_frame = frame.copy()

    for view_id, info in view_dict_by_id.items():
        # if info['frame_id'] not in car_info_dict_by_frame.keys():
        car_info_dict_by_frame[info['frame_id']] = {'result':info['result'],'frame':info['frame'],'frame_id':info['frame_id']}

    car_info_dict_by_frame = dict(sorted(car_info_dict_by_frame.items()))

    views = EMA(span = 10)

    for frame_id, frame_info in car_info_dict_by_frame.items():
        # car_info_dict_by_frame[frame_id]['result'][0] = correct_quarter_panel_base_fuel_tank_door(frame_info['result'][0])
        if frame_info['result'][0]['carpart']['view'] is None:
            car_info_dict_by_frame[frame_id]['view'] = 'None'
            continue
        car_info_dict_by_frame[frame_id]['view'] = views.add(frame_info['result'][0]['carpart']['view'])

    return car_info_dict_by_frame

def collect_result(car_info_dict_by_frame,hungarian,file_name):
    # flatten dict by frame 
    flatten_list = [info for _,info in car_info_dict_by_frame.items() if any([i in info['result'][0].keys() for i in ['dent','scratch']])]
    print('debug flatten list : ', len(flatten_list))
    flatten_list.sort(key=lambda x :x['frame_id'])

    for id,info in enumerate(flatten_list):
        print(info['frame_id'], ' : ',info['view'])
        result = info['result']
        # image_out_name = 'bin_'+str(bin)+'_frame_'+str(info['view'])+'_'+str(info['frame_id'])+'.jpg'
        # cv2.imwrite(images_video_path+'/'+image_out_name,info['frame'])
        result, damage_result = compare_masks(info['frame'],damage_filter,result[0])

        flatten_list[id]['result'] = result
        flatten_list[id]['damage_result'] = damage_result
        print('debug damage result : ',damage_result)

    scratch_cross_check = Cross_check_pair(hungarian)

    print('cross checking scratch damage ...')

    for id in range(len(flatten_list)):
        if not bool(flatten_list[id]['damage_result']):
            continue
        
        id_next = id + 1

        for idx in range(id_next,len(flatten_list)):
            if not bool(flatten_list[id_next]['damage_result']):
                id_next += 1
            else:
                break
        
        if id_next not in range(len(flatten_list)):
            continue
            
        print(' check pair id : ',id, id_next)
        flatten_list[id], flatten_list[id_next] = scratch_cross_check.cross_check(flatten_list[id],flatten_list[id_next])

    for id in range(len(flatten_list)):
        filter_carpart_by_view(flatten_list[id])  
    
    damages_by_frame = {}

    for id in range(len(flatten_list)):
        damages_by_frame[str(flatten_list[id]['frame_id'])+'_'+str(flatten_list[id]['view'])] = flatten_list[id]['damage_result']
        # print(flatten_list[id]['damage_result'])
    
    final_result = collect_final_result_after_cross_check(damages_by_frame)

    print(final_result)
    with open('video_out_new_pipeline/'+file_name+'.json', 'w', encoding='utf-8') as f:
        json.dump({'damaged_bin':damages_by_frame,'result':final_result}, f, ensure_ascii=False, indent=4)

    Path('video_out_new_pipeline/'+file_name+'/output').mkdir(parents=True,exist_ok=True)
    for info in flatten_list:
        draw_img = info['frame'].copy()
        cv2.imwrite('video_out_new_pipeline/'+file_name+'/'+str(info['frame_id'])+'.jpg',info['frame'])
        # print('*'*10)
        for cp_label,damage_list in info['damage_result'].items():
            for damage in damage_list:
                if damage[0] != 'scratch':
                    clr = (255,0,0)
                else:
                    clr = (0,0,255)

                # if not damage[-1] and damage[0]=='scratch':
                #     # print('skip : ',cp_label)
                #     continue

                # damage_mask = info['result'][damage[0]]['masks'][damage[2]].astype(bool)
                damage_box = info['result'][damage[0]]['bboxes'][damage[2]]
                cp_box = info['result']['carpart']['bboxes'][damage[3]]
                # print(cp_label,cp_box,damage_box)

                # draw_img[damage_mask] = draw_img[damage_mask]*0.5 + np.array([255,0,0])*0.5
                cv2.rectangle(draw_img,(damage_box[0],damage_box[1]),(damage_box[2],damage_box[3]),clr,2)

                if not damage[-1]:
                    cv2.line(draw_img,(damage_box[0],damage_box[1]),(damage_box[2],damage_box[3]),clr,2)

                cv2.rectangle(draw_img,(cp_box[0],cp_box[1]),(cp_box[2],cp_box[3]),(0,255,0),10)
                cv2.putText(draw_img,cp_label,(cp_box[0],cp_box[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,255,0),thickness=2)
        
        cv2.imwrite('video_out_new_pipeline/'+file_name+'/output'+'/'+str(info['frame_id'])+'.jpg',draw_img)

    return 

def evaluate_video(video_path):
    name = os.path.basename(video_path)
    name = name[:name.rfind('.')]

    cap = cv2.VideoCapture(video_path)  
    frame_w = int(cap.get(3))
    frame_h = int(cap.get(4))

    hungarian = Hungarian(frame_w,frame_h)

    frame_id = 0
    frame_queue = FrameQueue()
    car_info_dict_by_frame = {}

    while cap.isOpened():
        ret, frame = cap.read()

        if ret != True:
            break
        
        if frame_id % 5 == 0:
            result, car_info_flag = yolo_damage_inference(frame)

            if car_info_flag : 
                print('frame id !!!!!!!!!!!!: ',frame_id)
                frame_queue.enqueue({'frame':frame,'frame_id':frame_id})
                # run inference on framequeue
                # ...
                result = get_car_info(result, car_info_dict_by_frame,frame_queue,hungarian,name)
                car_info_dict_by_frame[frame_id]['result'] = result
                print('debug car info dict: ',len(car_info_dict_by_frame.keys()),' len frame queue : ',len(frame_queue.frame_id_list))

            # if frame_id % 3 == 0 : 
            frame_queue.enqueue({'frame':frame,'frame_id':frame_id})
            
            cv2.imwrite('video_tool/debug_new_pipeline.jpg',draw_result(result,frame.copy()))

        frame_id += 1
    

    car_info_dict_by_frame = verify_car_info_dict(car_info_dict_by_frame,hungarian)

    view_info = {}

    video_writer = cv2.VideoWriter('video_tool/out-cp-demo-'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),5,(frame_w,frame_h))

    for k,info in car_info_dict_by_frame.items():
        view_info[k] = str(info['result'][0]['carpart']['view'])
        image = draw_result(info['result'],info['frame'])
        video_writer.write(image)
    
    with open('video_tool/view_dict.json', 'w', encoding='utf-8') as f:
        json.dump(view_info, f, ensure_ascii=False, indent=4)

    collect_result(car_info_dict_by_frame,hungarian,name)
    
    return 

def main():
    video_files = glob.glob('video/video_test/20220728_123228.mp4')
    for video_file in video_files:
        evaluate_video(video_file) 

if __name__=='__main__':
    main()
