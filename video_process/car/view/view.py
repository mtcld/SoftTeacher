import cv2
from .three_d_car import color_segment
from video_process.car.view.util import compare_key
from video_process.car.view.detail_view import *
import time
from multiprocessing import Process, Value
from threading import Thread, current_thread


carpart2view = {'fli_fog_light': 1, 'sli_side_turn_light': 1, 'mirror': 0, 'fbu_front_bumper': 1,
                                   'bpn_bed_panel': -1, 'grille': 1, 'tail_gate': -1, 'mpa_mid_panel': 0, 'fender': 1,
                                   'hli_head_light': 1, 'car': 0, 'rbu_rear_bumper': -1,
                                   'door': 0,
                                   'lli_low_bumper_tail_light': -1,
                                   'hood': 1, 'hpa_header_panel': 1, 'trunk': -1, 'tyre': 0,
                                   'alloy_wheel': 0,
                                   'hsl_high_mount_stop_light': -1, 'rocker_panel+l': 0,'rocker_panel':0,  'qpa_quarter_panel': -1,
                                   'rpa_rear_panel': -1, 'rdo_rear_door': -1, 'tli_tail_light': -1,
                                   'fbe_fog_light_bezel': 1,'window':0}


def get_front_back_view(carparts):
    view_count = 0
    for carpart in carparts:
        carpart=carpart.split('+')[0]
        if  carpart not in carpart2view:
            continue
        view_count += carpart2view[carpart]
    return int(view_count / abs(view_count)) if view_count != 0 else 0

def get_left_right_view(carparts):
    view_count = 0
    for carpart in carparts:
        if '+'  not in carpart :
            continue 
        side=carpart.split('+')[-1]
        
        if 'l' in side:
            view_count +=1
        if 'r' in side:
            view_count -=1
            
    return int(view_count / abs(view_count)) if view_count != 0 else 0


class Car_View():
    def __init__(self,_3D_image_folder="image_3d"):
        self.process_3d_images(_3D_image_folder)

    def get_part_view(self, part):
        return self.carpart_cates[part]

    def process_3d_images(self,_3D_image_folder):
        self._2d_image_path_lists = color_segment.get_2d_image_path_lists(_3D_image_folder)
        self._2d_carpart_center_lists = color_segment.get_carpart_center_list_in_all_2d_image(self._2d_image_path_lists)


    def estimate_car_view(self, m_2d_carpart_center_lists, real_carpart_center_list, files):
        # adjust x of grille 
        # print('ggggggggg',real_carpart_center_list)
        grilles_x=[real_carpart[1] for real_carpart in real_carpart_center_list if 'grille' in real_carpart[0]]
        min_x= np.mean(grilles_x)
        
        for  real_carpart in real_carpart_center_list :
            if 'grille' in real_carpart[0]:
                real_carpart[1] = min_x
        score_list = []
        consider_file=[]
        for i, (m_2d_carpart_center_list,file) in enumerate(zip(m_2d_carpart_center_lists,files)):
            if len(real_carpart_center_list) -len(m_2d_carpart_center_list) >5:
                continue 

            score=compare_key(real_carpart_center_list,m_2d_carpart_center_list) 
            
            if score is None:
                continue 
            score_list.append(score)
            consider_file.append(file)
        if len(consider_file)==0:
            print('view','')
            return ''
       
        consider_file = [file for file, _ in
                sorted(zip(consider_file, score_list), key=lambda pair:-pair[1])]

        print('view', consider_file[0])
        return consider_file[0]

    def estimate_detailview(self,label2poss,masks):
        return get_detail_view(label2poss,masks)

    def estimate_overview(self,label2poss,view):
        # get carpart list in projection image
        proje_image_path=''
        if view < 0:
            proje_image_path = self.estimate_car_view(self._2d_carpart_center_lists, label2poss,
                                                                self._2d_image_path_lists)

            if proje_image_path=='':
                return None,proje_image_path
            view = (proje_image_path.split(".png")[0]).split("_")[-1]
        # print('debug : ',view)
        return view,proje_image_path


