import sys
import os

from video_process.utils.mask import *
from video_process.carpart.filter_carpart import *
from video_process.carpart.processing import CarpartProcessing
from tqdm import tqdm 

carpart_post=CarpartProcessing()
 
class CarpartInfo():
    def __init__(self):
        pass

    def add_index2label(self,labels):
        new_labels = []
        for i, label in enumerate(labels):
            new_labels.append(label + ":" + str(i))
        return new_labels


    def sort_by_bbox(self,masks,labels,scores,bboxes):
        # sort masks and labels folow bbox
        # print('debug sort : ',bboxes)
        masks = [x for _, x in
                sorted(zip(bboxes, masks), key=lambda pair:-(pair[0][2]-pair[0][0]) * (pair[0][3]-pair[0][1]))]
        labels = [x for _, x in
                sorted(zip(bboxes, labels), key=lambda pair:-(pair[0][2]-pair[0][0]) * (pair[0][3]-pair[0][1]))]

        scores = [x for _, x in
                sorted(zip(bboxes, scores), key=lambda pair:-(pair[0][2]-pair[0][0]) * (pair[0][3]-pair[0][1]))]
        
        bboxes = [x for x in sorted(bboxes, key=lambda pair:-(pair[2]-pair[0]) * (pair[3]-pair[1]))]

        return masks,labels,scores,bboxes


    def add_carpart_info(self,files,pred_jsons):
        # exits_carpart=[]
        sigle_door_car=False
        double_door_car=False    
        carparts=[]
        # print('debug : ', len(files),len(pred_jsons) )
        for k, (file,pred_json) in tqdm(enumerate(zip(files,pred_jsons))):
            #print(pred_json)
            filter_overlap_carpart(pred_json['carpart']['labels'],pred_json['carpart']['masks'],pred_json['carpart']['scores'],pred_json['carpart']['bboxes'])     
            filter_carpart(pred_json)
            label2poss, scores, masks,bboxes, car_bbox,car_area = carpart_post.get_maincar_carparts(pred_json) 
            pred_jsons[k]['car']['main_car'] = car_bbox
            remove_wrong_unique_side_carpart(label2poss,masks,scores,bboxes)
            label2poss,labels,proje_image_path,view= carpart_post.add_side2capart(label2poss,masks)
            
            if not sigle_door_car or not double_door_car:
                sigle_door_car1,double_door_car1=check_one_door_car(labels)
                sigle_door_car=sigle_door_car or sigle_door_car1
                double_door_car=double_door_car or double_door_car1
                if (sigle_door_car) :
                    print('this image has single door ')    
            if view is not None:
                # remove carpart by view
                # remove_carpart_from_view(labels, masks, scores,bboxes)
                only_keep_roof(labels, masks, scores,bboxes)



            masks,labels,scores,bboxes=self.sort_by_bbox(masks,labels,scores,bboxes)
            contours= get_contours(masks)
            egde_mask = masks_edge(contours,masks)

            for score,label in zip(scores,labels):
                if score > 0.6:
                    carparts.append(label)

            # if len(labels) > 0 and  ('c_90_90' in proje_image_path or 'c_90_270' in proje_image_path):
            #     view=None 
            pred_jsons[k]['carpart']={"labels": labels, "masks": masks,"scores":scores,"bboxes":bboxes,\
                "contours":contours,"edge_mask": egde_mask,'view':view,"totaled_info": {'view_img':proje_image_path,'car_bbox': car_bbox,
                                                                                                            'label2pos': label2poss},\
                                                                                                                "car_area":car_area}

        if sigle_door_car and not(double_door_car):
            print('*************************single door per side ')
            pred_jsons=adjust_label_on_sigle_door_car(pred_jsons)
            
            for pred_json in pred_jsons:
                pred_json['carpart']['single_door']= True

        for pred_json in pred_jsons:
            pred_json['carpart']['totaled_info']['carparts']= set(carparts)
        
        ## 
        # TODO : add checking fuel tank door + fender check here
        for pred_json in pred_jsons:
            pred_json = correct_quarter_panel_base_fuel_tank_door(pred_json)

        return pred_jsons



