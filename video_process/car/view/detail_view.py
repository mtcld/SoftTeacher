import cv2
import numpy as np
from video_process.car.view.mask import expend_mask
from video_process.car.view.recognize_unclear_view import check_unclear_view


detail_case=[['door','fender'],['door','qpa_quarter_panel'],['door','mirror'],\
['hli_head_light','hood','fbu_front_bumper'],['door','door','handle'],['door','door','handle','handle']]
def check_detail_case(labels):
    if len(labels) >9 :
        return False

    if labels.count('door') >=1 and (labels.count('fender') ==1 or labels.count('qpa_quarter_panel')==1 or labels.count('mirror')==1):
        return True
    
    if labels.count('hli_head_light') ==1 and (labels.count('hood') ==1 or labels.count('fbu_front_bumper')==1):
        return True

    if labels.count('door') ==2 and labels.count('handle') ==1:
        return True

    if labels.count('door') ==2 and (labels.count('handle') ==2) and (labels.count('qpa_quarter_panel') + labels.count('fender') + labels.count('mirror') ==0):
        return True
    return False

def get_detail_view(real_carpart_center_list,carpart_masks):

    view=-1
    raw_labels=[real_carpart[0] for real_carpart in real_carpart_center_list ] 
    scores = [real_carpart[-1] for real_carpart in real_carpart_center_list ] 
    
    if check_unclear_view(raw_labels,scores):
        return None

    if not check_detail_case(raw_labels):
        return view

    labels2info={}
    for real_carpart in real_carpart_center_list:
        if real_carpart[0] not in labels2info:
            labels2info[real_carpart[0]]=[]
        labels2info[real_carpart[0]].append(real_carpart[1:])

    #  filter 
    if 'fender' in labels2info and 'qpa_quarter_panel' in labels2info:
        fender_score=labels2info['fender'][0][2]
        quater_panel_score=labels2info['qpa_quarter_panel'][0][2]
        
        if fender_score > quater_panel_score:
            labels2info.pop('qpa_quarter_panel')
        else:
            labels2info.pop('fender')
    
    if 'fender' in labels2info and 'door' in labels2info:
        return fender_view(raw_labels,carpart_masks,labels2info)

    if 'qpa_quarter_panel' in labels2info and 'door' in labels2info:
        return quater_panel_view(raw_labels,carpart_masks,labels2info)

    if raw_labels.count('door') ==2 and raw_labels.count('handle')==1 :
        return single_handle_view(raw_labels,carpart_masks,labels2info)

    if 'mirror' in labels2info and 'door' in labels2info:
        return mirror_view(raw_labels,carpart_masks,labels2info)

    if raw_labels.count('door') ==2 and raw_labels.count('handle') >1:
        return multil_handel_view(raw_labels,carpart_masks,labels2info) 
    return -1    


def fender_view(labels,carpart_masks,labels2info):
    if 'rocker_panel' in labels2info and len(labels2info) <4:
        rocker_panel_mask=[expend_mask(mask) for i,mask in enumerate(carpart_masks) if 'rocker_panel'  in  labels[i]]
        fender_mask= [expend_mask(mask) for i,mask in enumerate(carpart_masks) if 'fender'  in  labels[i]]

        overlap = cv2.countNonZero(cv2.bitwise_and(fender_mask[0], rocker_panel_mask[0]))

        if overlap ==0 :
            return None 

    fender_X= labels2info['fender'][0][0] 
    door_X = labels2info['door'][0][0]

    if fender_X > door_X:
        view=50
    else:
        view=330
    return view 

def mirror_view(labels,carpart_masks,labels2info):
    if   'handle' not in labels2info:
        return None
    mirror_X= labels2info['mirror'][0][0] 
    door_X = labels2info['handle'][0][0]

    if mirror_X > door_X:
        view=50
    else:
        view=330
    
    return view

def single_handle_view(labels,carpart_masks,labels2info):
    handle_mask=[mask for i,mask in enumerate(carpart_masks) if 'handle'  in  labels[i]]
    door_masks=[mask for i,mask in enumerate(carpart_masks) if 'door'  in  labels[i]]

    door1_X= labels2info['door'][0][0]
    door2_X= labels2info['door'][1][0]

    #  check whether handle is front 

    door1_overlap = cv2.countNonZero(cv2.bitwise_and(door_masks[0], handle_mask[0]))
    door2_overlap = cv2.countNonZero(cv2.bitwise_and(door_masks[1], handle_mask[0]))
    if (door1_overlap - door2_overlap)*(door1_X  - door2_X) >0 :
        return  50
    else:
        return 330

def multil_handel_view(labels,carpart_masks,labels2info):
    handle_mask=[mask for i,mask in enumerate(carpart_masks) if 'handle'  in  labels[i]]
    door_masks=[mask for i,mask in enumerate(carpart_masks) if 'door'  in  labels[i]]

    door1_X= labels2info['door'][0][0]
    door2_X= labels2info['door'][1][0]

    two_hanlde_center=np.mean(np.array(labels2info['handle']),axis=0)
    if two_hanlde_center[0] < 0.5:
        return 50
    else :
        return 330 

def quater_panel_view(labels,carpart_masks,labels2info):
    quater_panel_X= labels2info['qpa_quarter_panel'][0][0] 
    door_X = min(labels2info['door'][:])[0]
    if quater_panel_X < door_X:
        view=140
    else:
        view=220
    return view
