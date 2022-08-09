import cv2
import os

from video_process.utils.mask import *
from video_process.car.processing import CarProcessing
from video_process.carpart.relation import CarpartRelation
from video_process.car.view.view import Car_View
from video_process.carpart.side import get_side_carpart

car_pro=CarProcessing()
print('debug path 3d :',os.path.join(os.getcwd(), "image_3d/"))
car_view = Car_View("video_process/image_3d/")

class CarpartProcessing():
    def __init__(self):
        pass
    

    def get_maincar_of_carparts(self,carpart_pred,lagest_car_mask,lagest_car_bbox,overlap_thresh): 
        capart_list_infos = []
        scores =[]
        masks=[]
        bboxes=[]
        for i, cat_id in enumerate(carpart_pred['labels']):
            if (check_mask_inside_mask(carpart_pred['masks'][i],lagest_car_mask,thresh=overlap_thresh,check_max=False)):
                cx, cy = get_mask_center(carpart_pred['masks'][i])

                # box= carpart_pred['bboxes'][i]
                # cx = (box[0] +box[2]) /2.0
                # cy = (box[1] +box[3]) /2.0

                if (cx < 0):
                    continue
                info = [cat_id, (cx - lagest_car_bbox[0]) / lagest_car_bbox[2], # self.cat[cat_id]
                        (cy - lagest_car_bbox[1]) / lagest_car_bbox[3],carpart_pred['scores'][i]]
                capart_list_infos.append(info)
                scores.append(carpart_pred['scores'][i])
                masks.append(carpart_pred['masks'][i])
                bboxes.append(carpart_pred['bboxes'][i])
                
        return capart_list_infos,scores,masks,bboxes

    def process(self,pred_json):
        carpart_pred = pred_json['carpart']
        if (len(carpart_pred['masks']) == 0):
            return None,None
        all_mask=np.zeros_like(carpart_pred['masks'][0], np.uint8)
        
        area_list = []
        index_list=[]
        max_score= max(carpart_pred['scores'])
        thresh = min(0.75,max_score)

        for i, (mask,score) in enumerate(zip(carpart_pred['masks'],carpart_pred['scores'])):
            mask = mask.astype(np.uint8)
            if score >=thresh:
                area_list.append(cv2.countNonZero(mask))
                index_list.append(i)
            mask=expend_mask(mask)
            all_mask=cv2.bitwise_or(all_mask,mask)
        if len(index_list) ==0:
            return None,None
            
        return carpart_pred['masks'][index_list[np.argmax(area_list)]],all_mask

    def get_maincar_carparts(self,pred_json):

        largest_car_mask,all_car_mask=car_pro.process(pred_json)
        # cv2.imwrite('debug_2.jpg',(largest_car_mask*255).astype(np.uint8))
        largest_carpart_mask,all_carpart_mask=self.process(pred_json)

        if 'missing' in pred_json:
            car_mask,car_bbox,overlap_thresh=\
                self.review_largest_car(largest_car_mask,largest_carpart_mask,all_car_mask,\
                 all_carpart_mask,pred_json['missing']['masks'],pred_json['car']['labels'])
        else:
            car_mask,car_bbox,overlap_thresh=\
                self.review_largest_car(largest_car_mask,largest_carpart_mask,all_car_mask,\
                 all_carpart_mask,[],pred_json['car']['labels'])
        car_mask=fill_hole_mask(car_mask)

        # debug = (car_mask*255).astype(np.uint8)
        # cv2.rectangle(debug,())
        # cv2.imwrite('debug_2.jpg',debug)
        
        # file hole  inside car_mask
        capart_list_infos,scores,masks,bboxes= self.get_maincar_of_carparts(pred_json['carpart'], car_mask, car_bbox,overlap_thresh)
        
        return capart_list_infos,scores,masks,bboxes,car_bbox,cv2.countNonZero(car_mask)


    def review_largest_car(self,max_car_mask,max_carpart_mask,all_mask_car,all_carpart_mask,missing_masks,car_labels):
        if all_carpart_mask is None :
            return None,[0,0,0,0],0.6
        if (cv2.countNonZero(max_car_mask) >0) :
            #  add missing mask to car mask
            all_mask_car = add_mask2masks(all_mask_car, missing_masks)
            max_car_mask = add_mask2masks(max_car_mask, missing_masks)

        if  cv2.countNonZero(all_mask_car)==0:
            #  add missing mask to carparts mask
            all_carpart_mask = add_mask2masks(all_carpart_mask, missing_masks)
            largest_mask=get_largest_mask(all_carpart_mask)
            return largest_mask,get_bbox(largest_mask),0.6

        max_carpart_mask = max_carpart_mask.astype(np.uint8)
        over_mask = cv2.bitwise_and(max_car_mask, max_carpart_mask)
        over_area= cv2.countNonZero(over_mask)
        max_carpart_area= cv2.countNonZero(max_carpart_mask)
        
        if max_carpart_area >0 and over_area /max_carpart_area >0.2:
            # check only one car 
            if len(car_labels) ==1:
                return expend_mask(max_car_mask,thickness=26),get_bbox(max_car_mask),0.8
            return max_car_mask,get_bbox(max_car_mask),0.6
        else:
            iv_mask=cv2.bitwise_not(all_mask_car)
            car_mask=cv2.bitwise_and(iv_mask,all_carpart_mask)
            return car_mask,get_bbox(car_mask),0.6


    def add_side2capart(self,label2poss,masks):
        view= car_view.estimate_detailview(label2poss,masks)
        
        proje_image_path=''
        if view is  not None :
            view, proje_image_path = car_view.estimate_overview(label2poss,view)
        label2poss = get_side_carpart(label2poss, view)
        # print('debug view :',view)
        nlabel2pos={}
        labels=[]
        for info in label2poss:
            nlabel2pos[info[0]]=info[1:3]
            labels.append(info[0])
        return nlabel2pos,labels,proje_image_path,view

        
