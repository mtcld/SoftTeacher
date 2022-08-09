
import cv2
import numpy as np
import os
from video_process.car.view.mask import get_largest_mask,get_bbox,\
    get_mask_center,get_contours,create_mask,expend_mask,convex_mask


carpart_color_list = {'sli_side_turn_light': [[0.25, 0.5, 0.5]], \
                      'mirror': [[0.65, 0.5, 0.5], [0.5, 0.5, 0.5]], \
                      'fbu_front_bumper': [[0.15, 1, 1]], \
                      'grille': [[0.8, 1, 1]], \
                      'tail_gate': [[0.9, 0.5, 0.5]], \
                      'fender': [[0.3, 1, 1], [0.3, 0.5, 1]], \
                      'hli_head_light': [[0.5, 1, 1], [0.5, 0.5, 1]], \
                      'rbu_rear_bumper': [[0.15, 0.5, 0.5], [0.8, 0.3, 1]], \
                      'door': [[0.1, 0.8, 0.5], [0.2, 0.6, 0.5], [0.45, 1, 1], [0.45, 0.5, 0.5]], \
                      'hood': [[0, 1, 1]], \
                      'tyre': [[0.7, 1, 1], [0.7, 0.5, 0.5], [0.5, 0.45, 1], [0.3, 0.5, 0.5]], \
                      'alloy_wheel': [[0.9, 1, 1], [0.35, 1, 1], [0.6, 1, 1], [0.75, 1, 1]], \
                      'rocker_panel': [[0.75, 0.5, 0.5], [0.95, 1, 1]], \
                      'qpa_quarter_panel': [[0.4, 0.6, 0.5], [0.4, 1, 1]], \
                      'tli_tail_light': [[0.35, 0.5, 0.5], [0.6, 0.5, 0.8]],\
                      'fwi_windshield':[[0.0,0.0,0.0]],\
                      'roof':[[0.1,0.1,1]],\
                      'handle':[[0.2,0.2,1],[0.25, 0.35, 1]],\
                      'fli_fog_light':[[0.8, 0.3, 1],[0.4, 0.15, 1]] 
                      }

carpart_side_color_list = {'sli_side_turn_light+f': [[0.25, 0.5, 0.5]], \
                      'mirror+lf': [[0.65, 0.5, 0.5]],'mirror+rf': [[0.5, 0.5, 0.5]], \
                      'fbu_front_bumper+f': [[0.15, 1, 1]], \
                      'grille+f': [[0.8, 1, 1]], \
                      'tail_gate+b': [[0.9, 0.5, 0.5]], \
                      'fender+lf': [[0.3, 1, 1]],'fender+rf' :[[0.3, 0.5, 1]], \
                      'hli_head_light+lf': [[0.5, 1, 1]], 'hli_head_light+rf':[[0.5, 0.5, 1]], \
                      'rbu_rear_bumper+b': [[0.15, 0.5, 0.5], [0.8, 0.3, 1]], \
                      'door+lf': [[0.1, 0.8, 0.5]],'door+lb':[[0.2, 0.6, 0.5]],'door+rb':[[0.45, 1, 1]],'door+rf': [[0.45, 0.5, 0.5]], \
                      'hood+f': [[0, 1, 1]], \
                      'tyre+lf': [[0.7, 1, 1]],'tyre+rf': [[0.7, 0.5, 0.5]], 'tyre+rb':[[0.5, 0.45, 1]],'tyre+lb':[[0.3, 0.5, 0.5]], \
                      'alloy_wheel+lf': [[0.9, 1, 1]],'alloy_wheel+lb':[[0.35, 1, 1]],'alloy_wheel+rb':[[0.6, 1, 1]],'alloy_wheel+rf': [[0.75, 1, 1]], \
                      'rocker_panel+r': [[0.75, 0.5, 0.5]], 'rocker_panel+l': [[0.95, 1, 1]], \
                      'qpa_quarter_panel+lb': [[0.4, 0.6, 0.5]],'qpa_quarter_panel+rb':[[0.4, 1, 1]], \
                      'tli_tail_light+rb': [[0.35, 0.5, 0.5]],'tli_tail_light+lb':[[0.6, 0.5, 0.8]],\
                      'fwi_windshield+f':[[0.0,0.0,0.0]],\
                      'roof':[[0.1,0.1,1]],\
                      'fli_fog_light+rf':[[0.8, 0.3, 1]],'fli_fog_light+lf':[[0.4, 0.15, 1]]
                      }

are_max_list={'sli_side_turn_light+f': 925.0, 'tyre+rb': 27696.0, 'fbu_front_bumper+f': 50722.5, 'door+rb': 48327.0, 'rocker_panel+l': 10850.0, 'alloy_wheel+lb': 16438.5, 'alloy_wheel+rf': 15363.5,\
              'hli_head_light+lf': 4554.5, 'qpa_quarter_panel+rb': 26272.5, 'fender+lf': 21781.0, 'tail_gate+b': 34980.0, 'mirror+lf': 2295.0, 'alloy_wheel+rb': 17404.0, 'fender+rf': 25704.5, \
              'qpa_quarter_panel+lb': 29738.0, 'hli_head_light+rf': 4831.0, 'tli_tail_light+rb': 8239.0, 'alloy_wheel+lf': 16243.5, 'hood+f': 40387.0, 'grille+f': 5442.5, 'mirror+rf': 2351.0,\
              'rocker_panel+r': 10857.5, 'tyre+lb': 29466.0, 'door+rf': 56199.5, 'tli_tail_light+lb': 8919.5, 'tyre+lf': 27138.0, 'rbu_rear_bumper+b': 62255.5, 'tyre+rf': 26081.5, 'door+lb': 49916.0,\
              'door+lf': 53241.0,'roof':55473.5,'fwi_windshield+f':24431.5,'fli_fog_light+rf':2001,'fli_fog_light+lf':2001}

def get_area_max_contour(contours):
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    if (len(areas) == 0):
        return 0, 0
    return max(areas), np.argmax(areas)


def segment_carpart(img, min_hsv, max_hsv):
    nemo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_nemo, min_hsv, max_hsv)
    return mask


def process_grille(mask):
    h,w=mask.shape[:2]
    conts =get_contours([mask])[0]
    
    new_mask=[]
    for cont in conts:
        if cv2.contourArea(cont) >170:
            new_mask.append(create_mask([[cont]],w,h))

    return  new_mask


def check_carpart_in_image(img, value, carpart_name,check_side=False):
    min_hsv = (max(0, int(value[0] * 180.0 - 2)), max(0, int(value[1] * 255.0 - 10)), max(0, int(value[2] * 255.0 - 5)))
    max_hsv = (
    min(180, int(value[0] * 180.0 + 2)), min(255, int(value[1] * 255.0 + 10)), min(255, int(value[2] * 255.0 + 5)))
    mask = segment_carpart(img, min_hsv, max_hsv)

    if (carpart_name == 'car'):
        mask = cv2.bitwise_not(mask)
    list_mask=[]
    if 'grille' in carpart_name:
        list_mask=process_grille(mask)
    else:
        mask=get_largest_mask(mask)
        list_mask =[mask]
    for i in range(len(list_mask) -1,-1,-1):
        mask =list_mask[i]
        area_mask=cv2.countNonZero(mask)
        if(area_mask<  170):
            list_mask.pop(i)
    return list_mask

def get_car_bbox(img):
    mask= check_carpart_in_image(img, (0, 0, 0.24), "car")[0]
    left_bbox, top_bbox, bbox_width, bbox_height = get_bbox(mask)
    return left_bbox, top_bbox, bbox_width, bbox_height


def get_center_carpart_list(image,check_side):
    carpart_center_list = []
    x, y, r_w, r_h = get_car_bbox(image)

    if(check_side):
        carpart_list=carpart_side_color_list
    else :
        carpart_list =carpart_color_list
    for carpart_name, colors in carpart_list.items():
        for color in colors:
            masks= check_carpart_in_image(image, color, carpart_name,check_side)
            for mask in masks:
                cx, cy = get_mask_center(mask)
                info = [carpart_name, (cx - x) / r_w, (cy - y) / r_h]
                carpart_center_list.append(info)
    return carpart_center_list


def get_carpart_center_list_in_all_2d_image(files,check_side=False):
    carpart_center_lists = []
    for path in files:
        image = cv2.imread(path)
        carpart_center_list = get_center_carpart_list(image,check_side)
        carpart_center_lists.append(carpart_center_list)
    return carpart_center_lists


def get_2d_image_path_lists(image_folder):

    files = []
    for r, d, f in os.walk(image_folder):
        for file in f:
            if ('.jpg' in file) or ('.png' in file):  # todo : not all the image extension
                files.append(os.path.join(r, file))
    return files

def get_all_projec_mask_labels(image_path,check_side):
   
    image=cv2.imread(image_path)
    view=image_path.split('_')[-1].split('.')[0]
    masks= []
    labels=[]
    x, y, r_w, r_h = get_car_bbox(image)
    if(check_side):
        carpart_list=carpart_side_color_list
    else :
        carpart_list =carpart_color_list

    for carpart_name, colors in carpart_list.items():
        if ((0 <= int(view)<=30) or (330 <= int(view)<=350)) and 'rear_bumper' in carpart_name:
            continue
        if 70 <= int(view)<=310 and 'fli_fog_light+rf' in carpart_name:
            continue
        if 0 <= int(view)<=40 and 'rbu_rear_bumper' in carpart_name:
            continue

        for color in colors:
            list_masks= check_carpart_in_image(image, color, carpart_name,check_side)
            
            for mask in list_masks:
                masks.append((mask[y:y+r_h,x:x+r_w]).astype(np.uint8))
                labels.append(carpart_name)

    if 'fender+rf' in labels and 'hli_head_light+rf' in labels :
        f_i = labels.index('fender+rf')
        hl_i= labels.index('hli_head_light+rf')
        masks[f_i] = cv2.bitwise_or(masks[f_i],expend_mask(masks[hl_i]))
    
    if 'fender+lf' in labels and 'hli_head_light+lf' in labels :
        f_i = labels.index('fender+lf')
        hl_i= labels.index('hli_head_light+lf')
        masks[f_i] = cv2.bitwise_or(masks[f_i],expend_mask(masks[hl_i]))

    if 'fli_fog_light+rf' in labels:
        fl_i = labels.index('fli_fog_light+rf')
        masks[fl_i]=convex_mask(masks[fl_i])*255
    
    for i,(label, mask) in enumerate(zip( labels,masks)):
        if 'front_bumper' in label:
            for label1, mask1 in zip(labels,masks):
                if 'grille' in label1 or 'fog_light' in label1 or 'flp_front_license_plate' in label1:
                    mask1=cv2.bitwise_not(mask1)
                    mask=cv2.bitwise_and(mask, mask1)   
            masks[i] =mask
    return masks,labels,image[y:y+r_h,x:x+r_w]


