import enum
from webbrowser import get
from video_process import Hungarian_matching
from video_process.SuperGluePretrainedNetwork.models_matching.matching import Matching
from video_process.SuperGluePretrainedNetwork.models_matching.utils import read_image
from video_process.Hungarian_matching import Hungarian

from mmdet.apis import init_detector, inference_detector, show_result_pyplot

from video_process.carpart.segment import CarpartInfo

import cv2
import numpy as np
import json
import torch
from sklearn.cluster import KMeans

from video_process.damage import Damage
from video_process.damage.scratch import Scratch

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

cp_model = init_detector('checkpoint/carpart_rear_exp_2/carpart_rear.py',
                      'checkpoint/carpart_rear_exp_2/epoch_29.pth',device='cuda:0')

car_model = init_detector('thirdparty/mmdetection/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
                          'checkpoint/coco_pretrain/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth',device='cuda:0')

scratch_model = init_detector('checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/swa-scratch-copy-paste-HSV-LSJ.py',
                        'checkpoint/scratch-cp-exp5-HSV-LSJ-SWA/best_bbox_mAP.pth',device='cuda:0')

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

def get_model_prediction(image):
    out = {}
    out['car'] = car_model_inference(car_model,image)
    out['carpart'] = carpart_model_inference(cp_model,image,0.5)
    
    return [out]

def get_key_points(img1,img2):
    image0, inp0, scales0 = read_image(img1, device, [1072, 1072], 0, False)
    image1, inp1, scales1 = read_image(img2, device, [1072, 1072], 0, False)

    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid1 = matches > -1
    valid2 = conf > 0.4
    valid = np.logical_and(valid1,valid2)
    # print(valid)
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    return mkpts0,mkpts1

def get_label(label):
    return label[:label.find('+')]

def filter_keypoints_by_carpart(kp1,mask1,kp2,mask2):
    # print()
    valid1 = [mask1[int(p[1]),int(p[0])] for p in kp1]
    valid2 = [mask2[int(p[1]),int(p[0])] for p in kp2]

    # print('number kp in carpart : ',sum(valid1),sum(valid2))
    if sum(valid1) < 100 or sum(valid2) < 100:
        valid = np.logical_or(valid1,valid2)
    else:
        valid = np.logical_and(valid1,valid2)

    return kp1[valid],kp2[valid]


def show_matches(image_pair,key_pairs=None,target_dim=800.0,path_out='result/result.jpg',):
    h1, w1 = image_pair[0].shape[:2]
    h2, w2 = image_pair[1].shape[:2]

    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2
        scale_to_fit = target_height / h1
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [
            target_w1, 0]

    target_1, target_2, scale1, scale2, offset = resize_horizontal(h1, w1, h2, w2, target_dim)

    im1 = cv2.resize( image_pair[0], target_1, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(image_pair[1], target_2, interpolation=cv2.INTER_AREA)

    h1, w1 = target_1[::-1]
    h2, w2 = target_2[::-1]

    vis = np.ones((max(h1, h2), w1 + w2, 3), np.uint8) * 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:w1 + w2] = im2

    if key_pairs is not None:
        p1 = [np.int32(k * scale1) for k in key_pairs[0]]
        p2 = [np.int32(k * scale2 + offset) for k in key_pairs[1]]

        li_color = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
        kk = 0
        
#         count  = 0
        for (x1, y1), (x2, y2) in zip(p1, p2):
            vis = cv2.circle(vis, (x1, y1), 5, li_color[kk % len(li_color)], 2)
            vis = cv2.circle(vis, (x2, y2), 5, li_color[kk % len(li_color)], 2)
            # cv2.line(vis, (x1, y1), (x2, y2), li_color[kk % len(li_color)], 2)
            # for i in range(2):
            #     for j in range(2):
            #         cv2.line(vis, (x1 -10*((-1)**i), y1-10*((-1)**j)), (x2-10*((-1)**i), y2 -10*((-1)**j)), li_color[kk % len(li_color)], 2)

            kk += 1
#             if kk==10 :
#                 break

    return vis

def make_coordinate(hungarian,kp1,kp2,space = 20):
    if len(kp1) <= space:
        return kp1,kp2
    
    kmean = KMeans(n_clusters=space,max_iter=50).fit(kp1)
    centers = kmean.cluster_centers_.astype(np.int32)
    # print('iterations : ',kmean.n_iter_)

    valid = hungarian.matching_points(centers,kp1)

    return kp1[valid],kp2[valid]

def main():
    damage = Damage()
    damage = Scratch(damage)

    img1 = cv2.imread('video_out/20220504_142734_bin_5/bin_15_0_frame_79.jpg')
    img2 = cv2.imread('video_out/20220504_142734_bin_5/bin_15_1_frame_76.jpg') 
    # img1 = cv2.imread('video_out/FBAI_Car_01_bin_5/bin_24_0_frame_123.jpg')
    # img2 = cv2.imread('video_out/FBAI_Car_01_bin_5/bin_24_1_frame_121.jpg')

    h,w,_ = img1.shape
    hungarian = Hungarian(w,h)

    carpart_info = CarpartInfo()

    result1 = get_model_prediction(img1)
    result1 = carpart_info.add_carpart_info(['krug/'],result1)
    result1 = damage_model_inference(scratch_model,result1,img1,score_thre=0.6)

    # print(result1[0]['scratch'])

    print('')

    for idx,b in enumerate(result1[0]['carpart']['bboxes']):
        # label_carpart = get_label(result1[0]['carpart']['labels'][idx])
        if any([i in result1[0]['carpart']['labels'][idx] for i in ['door+rf']]):
            # if tmp_track_carpart_flag[label_carpart] and track_carpart_flag[label_carpart]:
            # roi = [[int(b[0]),int(b[1])],[int(b[2]),int(b[1])],[int(b[2]),int(b[3])],[int(b[0]),int(b[3])]]
            # roi_list.append([result[0]['carpart']['labels'][idx],roi,idx])
            # tracking_flag = True

            #draw detection info
            cv2.rectangle(img1,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
            # score = round(result[0]['carpart']['scores'][idx],2)
            cv2.putText(img1,result1[0]['carpart']['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,0,255),thickness=2)
            mask1 = result1[0]['carpart']['masks'][idx]
    
    # 

    result2 = get_model_prediction(img2)
    result2 = carpart_info.add_carpart_info(['krug/'],result2)
    result2 = damage_model_inference(scratch_model,result2,img2,score_thre=0.6)

    for idx,b in enumerate(result2[0]['carpart']['bboxes']):
        # label_carpart = get_label(result2[0]['carpart']['labels'][idx])
        if any([i in result2[0]['carpart']['labels'][idx] for i in ['door+rf']]):
            # if tmp_track_carpart_flag[label_carpart] and track_carpart_flag[label_carpart]:
            # roi = [[int(b[0]),int(b[1])],[int(b[2]),int(b[1])],[int(b[2]),int(b[3])],[int(b[0]),int(b[3])]]
            # roi_list.append([result[0]['carpart']['labels'][idx],roi,idx])
            # tracking_flag = True

            #draw detection info
            cv2.rectangle(img2,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
            # score = round(result[0]['carpart']['scores'][idx],2)
            cv2.putText(img2,result2[0]['carpart']['labels'][idx],(b[0],b[3]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(0,0,255),thickness=2)
            mask2 = result2[0]['carpart']['masks'][idx]
    
    kp1,kp2 = get_key_points(img1,img2)

    kp1,kp2 = filter_keypoints_by_carpart(kp1,mask1,kp2,mask2)
    kp1,kp2 = make_coordinate(hungarian,kp1,kp2)

    cv2.imwrite('video_tool/demo-kp.jpg',show_matches([img1,img2],[kp1,kp2]))

    for p in kp1:
        # print(p)
        p = np.int32(p)
        # if mask1[p[1],p[0]]:
        cv2.circle(img1, (p[0], p[1]), 5, (255,0,0), 2)
    
    coord1 = np.array([int(p[0]) for p in kp1])
    
    for idm,mask in enumerate(result1[0]['scratch']['masks']):
        if np.logical_and(mask,mask1).sum() == 0:
            continue

        mask = mask.astype(bool)
        img1[mask] = 0.5*img1[mask] + 0.5*np.array([255,0,0])

        box = result1[0]['scratch']['bboxes'][idm]
        center = [int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
        cv2.circle(img1, center, 5, (255,0,255), 2)

        print('coor of center : ',(coord1<center[0]).sum())
    
    cv2.imwrite('video_tool/demo-kp1.jpg',img1)

    for p in kp2:
        # print(p)
        p = np.int32(p)
        # if mask1[p[1],p[0]]:
        cv2.circle(img2, (p[0], p[1]), 5, (255,0,0), 2)
    
    coord2 = np.array([int(p[0]) for p in kp2])
    
    for idm,mask in enumerate(result2[0]['scratch']['masks']):
        # print('intersect :',np.logical_and(mask,mask2).sum())
        if np.logical_and(mask,mask2).sum() == 0:
            continue
        mask = mask.astype(bool)
        img2[mask] = 0.5*img2[mask] + 0.5*np.array([255,0,0])

        box = result2[0]['scratch']['bboxes'][idm]
        center = [int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
        cv2.circle(img2, center, 5, (255,0,255), 2)

        print('coor of center : ',(coord2<center[0]).sum())

    cv2.imwrite('video_tool/demo-kp2.jpg',img2)

    

if __name__=='__main__':
    main()