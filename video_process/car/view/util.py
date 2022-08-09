import numpy as np
import cv2
from numpy import linalg as LA
from itertools import combinations
import math

#from car_damage.postprocess.utils.util import getAngle,check_carpart_in_list

def getAngle(list_points):
    a, b, c = list_points
    ba = a - b
    bc = c - b
    if (np.linalg.norm(ba) * np.linalg.norm(bc) == 0):
        return 1.57
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(min(cosine_angle, 0.99))
    return angle

def check_carpart_in_list(carpart,target):

    for t in target:
        if carpart in t:
            return True
    return False 


class Sorter:
    @staticmethod    
    def centerXY(xylist):
        x, y = zip(*(xylist.values()))
        l = len(x)
        return sum(x) / l, sum(y) / l  

    @staticmethod    
    def sortPoints(xylist):  
        cx, cy = Sorter.centerXY(xylist)
        
        xy_sorted = [k for k,v in sorted(zip(xylist.keys(),xylist.values()), key = lambda x: -math.atan2((x[1][1]-cy),(x[1][0]-cx)))]
        return xy_sorted

def sort_point(points):
    sortedPoints=Sorter.sortPoints(points)
    return sortedPoints

def gettriangle(list_points):
    p1, p2, p3 = list_points
    return np.asarray([getAngle([p2, p1, p3]), getAngle([p1, p2, p3]), getAngle([p1, p3, p2])])

def check_fog_light_real_img(real_labels):
    for real_label in real_labels:
        if 'fog_light' in real_label:
            return True
    return False 
    
def check_exit_pro_carpart(r_positions,p_positions,real_list_info,projection_info):
    for p_label,p_position in p_positions.items():
        if projection_info[0] in p_label and np.array_equal(p_position, np.asarray(projection_info[1:]), equal_nan=False) :
            old_score = np.exp(-LA.norm(r_positions[p_label] - np.asarray(projection_info[1:])) * 5)
            new_score = np.exp(-LA.norm(np.asarray(real_list_info[1:3]) - np.asarray(projection_info[1:])) * 5)

            if old_score <  new_score:
                r_positions[p_label] = np.asarray(real_list_info[1:3])
            
            return True, r_positions

    return False ,r_positions

def compare_key(real_carpart_center_list, m_2d_carpart_center_list):
    score = 0
    real_labels = [real_list_info[0] for real_list_info in real_carpart_center_list]
    pro_labels = [projection_info[0] for projection_info in m_2d_carpart_center_list]
    if 'grille' not in real_labels:
        score = score - 1
    for pro_label in pro_labels:
        if pro_label not in real_labels:
            if 'grille' in pro_label:
                continue
            score = score - 1

    for real_label in real_labels:
        if real_label not in pro_labels:
            score = score - 0.25

    score = score if score <-5 else score/2
    r_positions = {}
    p_positions = {}

    # is_fog_light= check_fog_light_real_img(real_labels)
    for i, real_list_info in enumerate(real_carpart_center_list):
        real_label = real_list_info[0].replace('fbe_fog_light_bezel','fli_fog_light') + str(i)
        real_position = np.asarray(real_list_info[1:3])

        max_score = -10000000

        for projection_info in m_2d_carpart_center_list:
            projection_info=projection_info.copy()
            
            # if 'fog_light' in projection_info[0] and (not is_fog_light ):
            #     projection_info[0]='grille'
            if (real_label == projection_info[0] + str(i)):
                projection_position = np.asarray(projection_info[1:])
                p_score = np.exp(-LA.norm(real_position - projection_position) * 5)

                if p_score < max_score:
                    continue             
                flag,r_positions=check_exit_pro_carpart(r_positions,p_positions,real_list_info,projection_info)
                if flag :

                    max_score = p_score
                    continue 
                
                r_positions[real_label] = real_position
                p_positions[real_label] = projection_position
                max_score = p_score
                # score = score + p_score

    intersect_labels = [label for label in r_positions.keys()]
    combine_list = list(map(list, combinations(intersect_labels, 3)))

    if len(combine_list) <2:
        return None

    for combine in combine_list[:56]:
        if check_carpart_in_list('tyre',combine) and check_carpart_in_list('alloy_wheel',combine):
            continue 
        r_p1 = r_positions[combine[0]]
        r_p2 = r_positions[combine[1]]
        r_p3 = r_positions[combine[2]]

        p_p1 = p_positions[combine[0]]
        p_p2 = p_positions[combine[1]]
        p_p3 = p_positions[combine[2]]
        
        score_cof =  check_triangle_flip({conbin:r_positions[conbin] for conbin in combine},\
            {conbin:p_positions[conbin] for conbin in combine})
        r_a1,r_a2,r_a3 = gettriangle([r_p1, r_p2, r_p3])
        p_a1,p_a2,p_a3 = gettriangle([p_p1, p_p2, p_p3])
        
        if score_cof <0  and max(r_a1,r_a2,r_a3) <2.62:
            score_cof =-2
        else :
            score_cof =abs(score_cof)

        # print(combine,score_cof,r_a1,r_a2,r_a3)  

        score = score + score_cof*(np.exp(-LA.norm(r_p1 - p_p1)*3 - np.abs(r_a1-p_a1)) +  np.exp(-LA.norm(r_p2 - p_p2)*3- np.abs(r_a2-p_a2)) +\
             np.exp(-LA.norm(r_p3 - p_p3)*3 - np.abs(r_a3-p_a3)))

    if math.isnan(score):
       return None
    return score


def check_triangle_flip(r_positions,p_positions):
    sorted_r=sort_point(r_positions)
    sorted_p=sort_point(p_positions)

    m_sorted_r=[sorted_r,[sorted_r[1],sorted_r[2],sorted_r[0]],[sorted_r[2],sorted_r[0],sorted_r[1]]]
    if sorted_p in m_sorted_r:
        return 1
    return -1


