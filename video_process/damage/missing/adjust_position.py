import numpy as np
import cv2 
from car_damage.postprocess.utils.mask import draw_contour,get_mask_center,get_bbox,process_front_bumper_mask
from car_damage.postprocess.carpart.relation import CarpartRelation
from car_damage.postprocess.utils.mask import expend_mask,create_mask
from numpy import linalg as LA



def get_line_mask_intersect(mask,points):
    line_mask=np.zeros((mask.shape[:2] + (3,)),np.uint8)
    line_mask=cv2.line(line_mask, (points[0]), (points[1]), (2,2,2), thickness=1)
    line_mask = cv2.cvtColor(line_mask, cv2.COLOR_BGR2GRAY)
    ret, line_mask = cv2.threshold(line_mask, 0.5, 1, 0)
    intersect=cv2.bitwise_and(mask,line_mask)
    intersect_points = cv2.findNonZero(intersect)

    if intersect_points is not None and len(intersect_points) >0:
        intersect_points = sorted(intersect_points, key=lambda x: LA.norm(np.array(x[0]) - np.array(points[0])))
        return intersect_points[-1][0].tolist()

    return None
def get_missing_point_on_projection_img(real_points,proj_mask):
    max_di= max(proj_mask.shape[:2])
    cx,cy=get_mask_center(proj_mask) 
    _,mask=draw_contour(proj_mask,thickness=2)   
    pro_points=[]
    
    for i in range(2):
        if real_points[i] is None:
            pro_points.append(None)
            continue
        vt = np.array(real_points[i]) -np.array(real_points[-1])
        x = cx + max_di*vt[0]
        y = cy + max_di*vt[1]

        intersect_point=get_line_mask_intersect(mask,[(cx,cy),(x,y)])
        pro_points.append(intersect_point)


    pro_points.append([cx,cy])
    return pro_points


def get_position_delta(proj_masks,proj_labels,missing_relations,carpart_infos):    
    # real_list=[]
    # pro_list=[]
    pos_deltas=[]
    for proj_mask, proj_label in zip(proj_masks, proj_labels):
        if proj_label in missing_relations:
            if 'front_bumper' in proj_label:
                
                for label1, mask1 in zip(proj_labels,proj_masks):
                    if 'grille' in label1 or 'fog_light' in label1 or 'flp_front_license_plate' in label1:
                        mask1=expend_mask(mask1)
                        proj_mask=cv2.bitwise_or(proj_mask, mask1)
                
                r_view=None
                # if int(carpart_infos['view'])<20 or int(carpart_infos['view'])==350: 
                    
                if 'fbu_front_bumper+f' in carpart_infos['labels']:
                    front_bum_mask = [mask for label,mask in zip(carpart_infos['labels'],carpart_infos['masks']) if 'fbu_front_bumper+f' in label ][0]
                    _,r_view,_ = process_front_bumper_mask(carpart_infos['labels'],carpart_infos['masks'],front_bum_mask)
                if  r_view is not None:       
                    proj_mask,_,_=process_front_bumper_mask(proj_labels,proj_masks,proj_mask,r_view)

            real_points = missing_relations[proj_label]

            if real_points.count(None)==2:
                pro_x,pro_y,pro_w,pro_h=get_bbox(proj_mask)
                pro_missing_x,pro_missing_y= np.array(real_points[-1])* np.array([pro_w,pro_h])
                pos_deltas.append(np.array([pro_missing_x+pro_x,pro_missing_y+pro_y])-np.array(real_points[2]))
                continue

            
            pro_points=get_missing_point_on_projection_img(real_points,proj_mask)
            
            for i in range(len(pro_points)):
                if pro_points.count(None) <2 and i >1:
                    continue
                if real_points[i] is None or pro_points[i] is None:
                    continue
                # real_list.append(real_points[i])
                # pro_list.append(pro_points[i])
                pos_deltas.append(np.array(pro_points[i])-np.array(real_points[i]))
                
    # real_list=np.array(real_list).reshape((-1, 1, 2)).astype(np.int32)
    # pro_list=np.array(pro_list).reshape((-1, 1, 2)).astype(np.int32) 

    delta = np.mean(np.array(pos_deltas), axis=0)
    # return delta,real_list,pro_list
    return delta
def contour_tran(contours,delta):
    new_contours=[]
    for cont in contours:
        new_point = []
        for point in cont:
            new_point.append(np.array([point[0] + delta]))
        new_contours.append(np.array(new_point).reshape((-1, 1, 2)).astype(np.int32))

    return new_contours

def adjust_position_mask(damage_mask,carpar_bbox,delta):
    x, y, w, h=carpar_bbox
    new_contours=[]
    old_contours=[]
    damage_mask = damage_mask[y:y + h, x:x + w]
    _, contours,_ = cv2.findContours(damage_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_contour=contour_tran(contours,delta)
    new_contours.append(new_contour)
    old_contours.append(contours)

    damage_mask=create_mask(new_contours, w, h)

    return damage_mask,new_contours,old_contours

def missing_relation(carpart_infos,miss_mask):
    cr = CarpartRelation(carpart_infos['masks'], carpart_infos['labels'],\
        carpart_infos['totaled_info']['car_bbox'], [miss_mask])
    missing_rela = {}
    for label,points in cr.get_neighbor().items():
        if ('grille' in label):
            continue
        label=label.split(':')[0]
        missing_rela[label]=points
    return missing_rela
