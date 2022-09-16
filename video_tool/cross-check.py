from calendar import c
from distutils.log import info
from pathlib import Path
import pickle
from video_process import damage
from video_process.damage import Damage
from video_process.damage.scratch import Scratch
from video_process.damage.dent import Dent

from video_process.Hungarian_matching import Hungarian
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector as st_init_detector

import numpy as np
import json
import torch
import cv2
import os
import glob

from video_process.SuperGluePretrainedNetwork.models_matching.matching import Matching
from video_process.SuperGluePretrainedNetwork.models_matching.utils import read_image

from sklearn.cluster import KMeans


torch.set_grad_enabled(False)

device = 'cuda:0'

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)

scratch_model = init_detector('checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/swa-scratch-copy-paste-HSV-LSJ.py',
                        'checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/best_bbox_mAP.pth',device='cuda:0')

dent_model = st_init_detector('checkpoint/19_02_2022/dent/dent_mask.py',
                        'checkpoint/19_02_2022/dent/iter_207000.pth',device='cuda:0')

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

    # if len(pred_json) == 1 : 
    #     print('debug : ',len(pred_json),pred_json[0].keys())
    # else:
    #     print('debug else : ',pred_json.keys())
    
    pred_json[0][model.CLASSES[0]] = {"labels": [model.CLASSES[l] for l in pred_labels], "scores": pred_scores.tolist(), "masks": [s.astype(np.uint8) for s in pred_segms],'bboxes':out_pred}

    return pred_json

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

def collect_final_result(damaged_by_bin_json):
    confirm = {}
    confirm2 = {}
    for id,(k,value) in enumerate(damaged_by_bin_json.items()):
    #     print(value)
        for cp, damages in value.items():
            for d in damages:
                if d[0] == 'scratch':
                    if d[1] > 0.61 and (d[-1] or any([i in cp for i in ['mirror','rocker_panel','hood']])):
                        label = cp+'_'+d[0]
                        if label not in confirm:
                            confirm[label] = 1
                        else:
                            confirm[label] += 1


                    if d[1] > 0.61 :
                        label = cp+'_'+d[0]
                        if label not in confirm2:
                            confirm2[label] = 1
                        else:
                            confirm2[label] += 1
                else:
                    if id % 3 == 0:
                        continue
                    label = cp+'_'+d[0]
                    if label not in confirm:
                        confirm[label] = 1
                    else:
                        confirm[label] += 1
    
    print('uncheck result  : ',len(confirm2.values()),confirm2)

    for k in list(confirm.keys()):
        if 'rocker_panel' in k and confirm[k] == 1:
            del confirm[k]
    return confirm

class Cross_check_pair():
    def __init__(self,hungarian):
        self.hungarian = hungarian
    
    def setup(self,info1,info2):
        self.info1 = info1
        self.info2 = info2

        self.draw1 = info1['frame'].copy()
        self.draw2 = info2['frame'].copy()

        self.kp1,self.kp2= self.get_matching_key_points(self.info1['frame'],self.info2['frame'])
    
  
    def get_matching_key_points(self,img1,img2):
        image0, inp0, scales0 = read_image(img1, device, [1072, 1072], 0, False)
        image1, inp1, scales1 = read_image(img2, device, [1072, 1072], 0, False)

        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid1 = matches > -1
        valid2 = conf > 0.3
        valid = np.logical_and(valid1,valid2)

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC,5.0)
        # self.transform_matrix = M

        return mkpts0,mkpts1
    
    def filter_keypoints_by_carpart(self,mask1,mask2):
        valid1 = [mask1[int(p[1]),int(p[0])] for p in self.kp1]
        valid2 = [mask2[int(p[1]),int(p[0])] for p in self.kp2]

        if sum(valid1) < 100 or sum(valid2) < 100:
            valid = np.logical_or(valid1,valid2)
        else:
            valid = np.logical_and(valid1,valid2)

        return self.kp1[valid],self.kp2[valid]

    def make_coordinate(self,mask1,mask2,space=20):
        kp1,kp2 = self.filter_keypoints_by_carpart(mask1,mask2)

        if len(kp1) < 20:
            return kp1,kp2
        
        kmean = KMeans(n_clusters=space,max_iter=50).fit(kp1)
        centers = kmean.cluster_centers_.astype(np.int32)

        valid = self.hungarian.matching_points(centers,kp1)
        kp1 = kp1[valid]
        kp2 = kp2[valid]

        return np.array([int(p[0]) for p in kp1]), np.array([int(p[0]) for p in kp2])
    
    # def estimate_bbox1(self,roi):
    #     pts = np.float32(roi).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,self.transform_matrix)

    #     return dst.astype(np.int32)
    
    def estimate_bbox2(self,roi_list):
        trackers = cv2.legacy.MultiTracker_create()

        for roi in roi_list:
            x1, y1, x2, y2 = roi
            tracker = cv2.legacy.TrackerCSRT_create()
            trackers.add(tracker,self.info1['frame'],[x1,y1,x2-x1,y2-y1])
        
        _,bboxes = trackers.update(self.info2['frame'])

        out = []
        for b in bboxes:
            b = np.int32(b)
            out.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

        return out

    def cross_check(self,info1,info2):
        def get_bbox(result,damage_list):
            roi_list = []
            idx_list = []

            for idx,d in enumerate(damage_list):
                if d[0] != 'scratch' :
                    continue

                b = result['scratch']['bboxes'][d[2]]
                roi = b

                roi_list.append(roi)
                idx_list.append(idx)

            return roi_list,idx_list
        
        self.setup(info1,info2)

        for carpart_label1,damage_list1 in self.info1['damage_result'].items():
            print('check car part ',carpart_label1)
            scratch_list1,ind_list1 = get_bbox(self.info1['result'],damage_list1)

            if len(scratch_list1) == 0:
                continue

            es_scratch_list1 = self.estimate_bbox2(scratch_list1)

            for b in scratch_list1:
                cv2.rectangle(self.draw1,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
            
            for b in es_scratch_list1:
                cv2.rectangle(self.draw2,(b[0],b[1]),(b[2],b[3]),(255,0,255),2)


            carpart_mask1 = self.info1['result']['carpart']['masks'][damage_list1[ind_list1[0]][3]]

            for carpart_label2,damage_list2 in self.info2['damage_result'].items():
                if carpart_label2 != carpart_label1:
                    continue

                scratch_list2,ind_list2 = get_bbox(self.info2['result'],damage_list2)

                if len(scratch_list2) == 0:
                    continue

                carpart_mask2 = self.info2['result']['carpart']['masks'][damage_list2[ind_list2[0]][3]]
                coord_kp1,coord_kp2 = self.make_coordinate(carpart_mask1,carpart_mask2)
                
                # print('image 2 ')
                coord_list2 = []
                for b in scratch_list2:
                    center = [int((b[0]+b[2])/2),int((b[1]+b[3])/2)]
                    coord = (coord_kp2<center[0]).sum()
                    coord_list2.append(coord)

                coord_list1 = []
                for b in scratch_list1:
                    center = [int((b[0]+b[2])/2), int((b[1]+b[3])/2)]
                    coord = (coord_kp1<center[0]).sum()
                    coord_list1.append(coord)
            
                self.info1,self.info2 = self.hungarian.matching_damages(carpart_label1,es_scratch_list1,ind_list1,coord_list1,self.info1,scratch_list2,ind_list2,coord_list2,self.info2)

                for b in scratch_list2:
                    cv2.rectangle(self.draw2,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
            # print('*'*200)  
            # cv2.imwrite('demo2.jpg',self.draw2)
            # cv2.imwrite('demo1.jpg',self.draw1)
                    
        return self.info1, self.info2

def main():
    damage_filter = Damage()
    damage_filter = Scratch(damage_filter)
    # damage_filter = Dent(damage_filter)

    video_datas = glob.glob('video_tool/*.pickle')
    # video_datas = glob.glob('video_tool/FBAI_Car_01.pickle')

    
    for file in video_datas:
        print(file)
        print('getting frame infomation ....')
        bin_dict = pickle.load(open(file,'rb'))
        name = os.path.basename(file)
        name = name[:name.rfind('.')]
        # bin_dict = pickle.load(open('video_tool/20220504_142734.pickle','rb'))
        # bin_dict = pickle.load(open('video_tool/IMG_5435.pickle','rb'))

        name = os.path.basename(file)
        name = name[:name.rfind('.')]

        flatten_list = []

        for bin, infos in bin_dict.items():
            for idx,info in enumerate(infos) :
                # print('debug id : ',idx)
                flatten_list.append(info)

        h,w,_ = flatten_list[0]['frame'].shape
        hungarian = Hungarian(w,h)

        flatten_list.sort(key=lambda x :x['frame_id'])

        damaged_by_frame = {}

        # inference scratch model
        print('inference damges model ...')
        for id,info in enumerate(flatten_list):
            # if id not in range(31,52+1):
            #     continue

            result = info['result'] 
            result = damage_model_inference(scratch_model,result,info['frame'],0.01)
            # result = damage_model_inference(dent_model,result,info['frame'],score_thre=0.35)
            # print(damage)
            result, damage_result = compare_masks(info['frame'],damage_filter,result[0])
            flatten_list[id]['result'] = result
            flatten_list[id]['damage_result'] = damage_result
            # damaged_by_frame[info['frame_id']] = damage_result
            print('damage reulst. id : ',id,' damage_result : ',damage_result)
        
        cross_check = Cross_check_pair(hungarian)
        print('cross check ...')
        for id in range(len(flatten_list)):
            # if id not in range(31,52+1):
            #     continue

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
                
            print('id check : ',id,id_next,flatten_list[id]['view'],flatten_list[id_next]['view'])

            flatten_list[id],flatten_list[id_next] = cross_check.cross_check(flatten_list[id],flatten_list[id_next])
        
        for id in range(len(flatten_list)):
            damaged_by_frame[flatten_list[id]['frame_id']] = flatten_list[id]['damage_result']
        
        #### draw result 
        out_path = 'video_out/'+name+'/scratch'
        Path(out_path).mkdir(parents=True,exist_ok=True)

        for info in flatten_list:
            draw_img = info['frame']
            # cv2.imwrite('video_out/'+name+'/'+str(info['frame_id'])+'.jpg',draw_img)
            # print('*'*10)
            for cp_label,damage_list in info['damage_result'].items():
                for damage in damage_list:
                    if damage[0] != 'scratch':
                        continue

                    if not damage[-1]:
                        # print('skip : ',cp_label)
                        continue

                    damage_mask = info['result']['scratch']['masks'][damage[2]].astype(bool)
                    damage_box = info['result']['scratch']['bboxes'][damage[2]]
                    cp_box = info['result']['carpart']['bboxes'][damage[3]]
                    # print(cp_label,cp_box,damage_box)

                    draw_img[damage_mask] = draw_img[damage_mask]*0.5 + np.array([255,0,0])*0.5
                    cv2.rectangle(draw_img,(damage_box[0],damage_box[1]),(damage_box[2],damage_box[3]),(255,0,0),2)

                    cv2.rectangle(draw_img,(cp_box[0],cp_box[1]),(cp_box[2],cp_box[3]),(0,255,0),10)
                    cv2.putText(draw_img,cp_label,(cp_box[0],cp_box[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,255,0),thickness=2)
            
            cv2.imwrite(out_path+'/'+str(info['frame_id'])+'.jpg',draw_img)

        final_result = collect_final_result(damaged_by_frame)

        with open('video_out/'+name+'_cross_check.json', 'w', encoding='utf-8') as f:
            json.dump({'detail':damaged_by_frame,'result':final_result}, f, ensure_ascii=False, indent=4)

        print(len(final_result.values()),final_result)

if __name__=='__main__':
    main()