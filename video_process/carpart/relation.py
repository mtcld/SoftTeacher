import cv2
import numpy as np
from video_process.utils.mask import expend_mask,get_mask_center,get_bbox,process_front_bumper_mask,fill_hole_mask

from video_process.utils.util import getAngle

def get_extreme_points(cnts):
    leftmost=(100000,0)
    rightmost=(-1000000,0)
    topmost = (0, 100000)
    bottommost = (0, -1000000)
    for cnt in cnts:
        new_leftmost=tuple(cnt[cnt[:, :, 0].argmin()][0])
        leftmost =new_leftmost if  new_leftmost[0] < leftmost[0] else leftmost

        new_rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        rightmost = new_rightmost if new_rightmost[0] > rightmost[0] else rightmost

        new_topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        topmost =new_topmost if  new_topmost[1] < topmost[1] else topmost

        new_bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        bottommost = new_bottommost if new_bottommost[1] > bottommost[1] else bottommost
       
    return leftmost,rightmost,topmost,bottommost


class CarpartRelation():
    def __init__(self,masks,labels,carbox,missings=None):
        car_x,car_y,car_w,car_h=carbox[:]
        self.masks=[mask[car_y:car_y+car_h,car_x:car_x+car_w] for mask in masks]
        self.labels=labels
        if missings is not None:
            missing = missings[0]
            for m in missings[-1:0:-1]:
                missing = cv2.bitwise_or(missing,m)
            self.missing=expend_mask(missing[car_y:car_y+car_h,car_x:car_x+car_w])


    def get_neighbor(self,):
        relation={}
        if len(self.missing)==0:
            return relation

        if self.missing is not None :
            for label, mask in zip(self.labels,self.masks):
                
                if 'fbu_front_bumper' in label:
                    for label1, mask1 in zip(self.labels,self.masks):
                        if  'grille' in label1 or 'fog_light' in label1 or 'flp_front_license_plate' in label1:
                            mask1=expend_mask(mask1)
                            mask=cv2.bitwise_or(mask, mask1)
                    or_h,or_w=mask.shape[:2]
                    mask = cv2.copyMakeBorder(mask, 0, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 1)
                    mask=fill_hole_mask(mask)
                    mask =mask[0:or_h,10:10+or_w]       
                intersection=cv2.bitwise_and(self.missing, expend_mask(mask,thickness=36))

                area_missing=cv2.countNonZero(self.missing)
                if area_missing==0:
                    continue
                area_intersect=cv2.countNonZero(intersection)
                if area_intersect==0:
                    continue
                
                if 'fbu_front_bumper' in label:
                    mask,r_view,center_bumper_x=process_front_bumper_mask(self.labels,self.masks,mask)
                
                if area_intersect/area_missing >0.8 :
                    offset_x=0
                    if 'fbu_front_bumper' in label and r_view is not None:
                        if not r_view :
                            offset_x=center_bumper_x
                    missing_x,missing_y= get_mask_center(self.missing)
                    x,y,w,h=get_bbox(mask)
                    info=[None,None,[missing_x+ offset_x ,missing_y],[(missing_x +offset_x-x)/w,(missing_y-y)/h ]]

                    relation[label]=info
                    continue

                else:
                    label_center= get_mask_center(mask)
                    _,i_conts, hierarchy = cv2.findContours(intersection, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    list_point=np.array(get_extreme_points(i_conts))
                    info=[]
                    max_angle=-1

                    for i in range(len(list_point)-1,0,-1):
                        for j in range(i-1,-1,-1):
                            angle=getAngle([list_point[i],label_center,list_point[j]])
                            if  angle> max_angle:
                                max_angle=angle
                                info=[list_point[i].tolist(),list_point[j].tolist(),label_center]
                    relation[label] =  info
        return relation 


                    




                    



