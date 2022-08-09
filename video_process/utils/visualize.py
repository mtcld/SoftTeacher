    
import cv2
import numpy as np
import os 

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    labels=np.arange(len(labels))
    colors = (labels[:, None]+1) * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array((colors % 255)).astype("uint8")
    return colors
def get_text_bbox(boxes, score_label):
    text_size = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
    x, y = boxes[:2]
    return [x, y - text_size[0][1], text_size[0][0], text_size[0][1]]  # format x,y,w,h

def check_2_bbox_overlap( bbox1, bbox2):
    bbox1_center = [bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0]
    bbox2_center = [bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0]

    if (abs(bbox1_center[0] - bbox2_center[0]) < (bbox1[2] + bbox2[2]) and abs(
            bbox1_center[1] - bbox2_center[1]) < (bbox1[3] + bbox2[3])):
        return True
    return False

def get_position_of_text(text_bbox, exited_text_bbox_list, width, height):

    x = min(text_bbox[0] + text_bbox[2], width) - text_bbox[2]
    y = max(text_bbox[1] + text_bbox[3], 0) + text_bbox[3]

    for i in range(20):
        new_text_bbox = [x, y - text_bbox[3], text_bbox[2], text_bbox[3]]
        overlap_y_list = [(old_text_bbox[1]) for old_text_bbox in exited_text_bbox_list if
                          (check_2_bbox_overlap(new_text_bbox, old_text_bbox) == True)]

        if (len(overlap_y_list) > 0):
            y = min(max(overlap_y_list) + 3 * text_bbox[3], height - 4)
        if (len(overlap_y_list) == 0 or y == height - 4):
            break
    return x, y, new_text_bbox

def get_part_name(car_part):
    token = str(car_part).split("_")
    if len(token) == 1 or len(token[0]) > 3:
        return " ".join(token)
    else:
        return " ".join(token[1:])

def draw_image(pre_json,img,car_flag,carpart_flag,act_cats):
    labels=pre_json['labels']
    bboxes=pre_json['bboxes']
    masks=pre_json['masks']
    scores=pre_json['scores']

    colors=compute_colors_for_labels(labels).tolist()
    template = "{}: {:.2f}"
    exited_text_bbox_list = []
    height, width = img.shape[:2]
    for label,bbox,mask,score,color in zip(labels,bboxes,masks,scores,colors):
        
        if car_flag==True and label !='car':
            continue
        #draw bbox
        bbox_int = bbox.astype(np.int32)
        img = cv2.rectangle(img, (bbox_int[0],bbox_int[1]), (bbox_int[2],bbox_int[3]), color, 1)

    for label,bbox,mask,score,color in zip(labels,bboxes,masks,scores,colors):
        # if 'door' not in label:
        #     continue 
        if (car_flag==True and label !='car'):
            continue
        # draw mask
        if carpart_flag:
            continue
        _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, color, 1)


    for label,bbox,mask,score,color in zip(labels,bboxes,masks,scores,colors):

        if car_flag==True and label !='car' :
            continue
        bbox_int = bbox.astype(np.int32)
        # draw label
        s = template.format(get_part_name(label), score)
        text_bbox = get_text_bbox(bbox_int, s)
        x, y, new_text_bbox = get_position_of_text(text_bbox, exited_text_bbox_list, width, height)
        exited_text_bbox_list.append(new_text_bbox)
        cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        for i,act_cat in enumerate(act_cats):
            cv2.putText(img, act_cat, (int(200), int(40+i*20)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
    return img


def prediction_visualize(pre_jsons,files,ouput_dir='data/output',act_cats=[]):
    os.makedirs(ouput_dir, exist_ok=True)        
    for img_path, pre_json in zip(files,pre_jsons):
        img=cv2.imread(img_path)
        seg_img_folder=os.path.join(ouput_dir, img_path.rsplit(".", 1)[0].split("/")[-1])
        os.makedirs(seg_img_folder, exist_ok=True)
        
        # save segment image
        for category, prediction in pre_json.items():
            draw_img=img.copy()

            if 'carpart' in category:
                draw_img=draw_image(prediction,draw_img,False,True,act_cats)
            elif 'car' in category:
                draw_img=draw_image(prediction,draw_img,True,False,act_cats)
            else:
                draw_img=draw_image(prediction,draw_img,False,False,act_cats)
            # seg_img_path = os.path.join(seg_img_folder, img_path.rsplit(".", 1)[0].split("/")[-2] +'_'+img_path.rsplit(".", 1)[0].split("/")[-1]+'_' +category + ".jpg")
            # cv2.imwrite(seg_img_path, np.concatenate((img, draw_img), axis=1))

            # print(seg_img_folder,prediction['labels'])
            seg_img_path = os.path.join(seg_img_folder, category + ".jpg")
            cv2.imwrite(seg_img_path, draw_img)
            

def visual_yolo_result(image_path,bboxes,confs,damage):
    damage_path='/'.join(image_path.split('/')[:-1]) +'/output/' +\
     image_path.split('/')[-1].split('.')[0] +'/'+damage +'.jpg'

    damage_imag=cv2.imread(damage_path)

    for bbox,conf  in zip(bboxes,confs):
        x1,y1,x2,y2=bbox
        cv2.rectangle(damage_imag,(x1,y1),(x2,y2),(255,0,255),1)
        cv2.putText(damage_imag, 'yolo'+str(conf.item()), (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)
    
    cv2.imwrite(damage_path,damage_imag)

    