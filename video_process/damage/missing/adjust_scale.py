
import cv2
import numpy as np 
from car_damage.postprocess.utils.mask import get_mask_center

def add_border2img(mask,cX,cY):
    i_h,i_w= mask.shape[:2]
    if cX > i_w/2 :
        mask = cv2.copyMakeBorder(mask, 0, 0 , 0, 2*cX - i_w, cv2.BORDER_CONSTANT, None, value = 0)

    else:
        mask = cv2.copyMakeBorder(mask, 0, 0 , -2*cX +i_w, 0, cv2.BORDER_CONSTANT, None, value = 0)

    if cY > i_h/2 :
        mask = cv2.copyMakeBorder(mask, 0, 2*cY -i_h , 0, 0, cv2.BORDER_CONSTANT, None, value = 0)

    else:
        mask = cv2.copyMakeBorder(mask,  -2*cY +i_h, 0 , 0,0, cv2.BORDER_CONSTANT, None, value = 0)

    return mask


def scale_mask(mask,center,ratio_scale,no_crop=False):
    cX,cY= center[:]
    i_h,i_w= mask.shape[:2]
    mask = add_border2img(mask,cX,cY)
    o_h,o_w =mask.shape[:2]  
    mask1=mask.copy()
    new_w  =int(o_w*ratio_scale)
    new_h  =int(o_h*ratio_scale)
    
    mask1=cv2.resize(mask1,(new_w,new_h))
    if no_crop:
        return mask1
    new_img=mask1[int(new_h/2-  cY) : int(new_h/2-  cY+ i_h) , int(new_w/2-  cX) : int(new_w/2-  cX+ i_w)]
    
    return new_img

def check_coincident_point(real_labels,real_masks,project_labels,proj_masks,missing_mask):

    coincident_points=0
    for real_label, real_mask in zip(real_labels,real_masks) :
        real_mask= cv2.bitwise_or(real_mask,missing_mask)
        for project_label,proj_mask in zip(project_labels,proj_masks):
            if real_label==project_label:
                coincident_points += cv2.countNonZero(cv2.bitwise_and(real_mask,proj_mask))

    return coincident_points


def search_max_scale(miss_center,proj_masks,project_labels,real_labels,real_masks,missing_mask):

    old_coin_points=-1

    for i,ratio_scale in enumerate(np.arange(1.0, 2.0, 0.1)):
        if i ==9:
            return  ratio_scale
        s_project_masks = [scale_mask(mask,miss_center,ratio_scale) for mask in proj_masks]
        coincident_points =check_coincident_point(real_labels,real_masks,project_labels,s_project_masks,missing_mask)
        
        if coincident_points > old_coin_points:
            old_coin_points=coincident_points
            continue 
        return ratio_scale -0.1


def adjust_scale_mask(proj_masks, proj_labels,real_masks,real_labels,damage): 
    cX,cY= get_mask_center(damage)
    if cX==-1 and cY==-1:
        return 1,[cX,cY],proj_masks
    ratio_scale= search_max_scale([cX,cY],proj_masks,proj_labels,real_labels,real_masks,damage)
    ratio_scale=max(1.0, ratio_scale)

    if 'fender+rf' in proj_labels and 'hli_head_light+rf' in proj_labels :
        f_i = proj_labels.index('fender+rf')
        hl_i= proj_labels.index('hli_head_light+rf')
        proj_masks[f_i] = cv2.bitwise_xor(proj_masks[f_i],proj_masks[hl_i])

    if 'fender+lf' in proj_labels and 'hli_head_light+lf' in proj_labels :
        f_i = proj_labels.index('fender+lf')
        hl_i= proj_labels.index('hli_head_light+lf')
        proj_masks[f_i] = cv2.bitwise_xor(proj_masks[f_i],proj_masks[hl_i])
        
    if ratio_scale ==1.0 :
        return ratio_scale,[cX,cY],proj_masks
        
    for i in range(len(proj_labels)-1,-1,-1):
        new_mask = scale_mask(proj_masks[i],[cX,cY],ratio_scale)
        origin_mask = scale_mask(proj_masks[i],[cX,cY],ratio_scale,no_crop=True)
        if 'fender' in proj_labels[i] and cv2.countNonZero(new_mask)/(cv2.countNonZero(origin_mask))<0.65:
            proj_labels.pop(i)
            proj_masks.pop(i)
            continue

        if cv2.countNonZero(new_mask)/(cv2.countNonZero(origin_mask)) < 0.45:
            proj_labels.pop(i)
            proj_masks.pop(i)

    return  ratio_scale,[cX,cY],[scale_mask(mask,[cX,cY],ratio_scale) for mask in proj_masks]

# for contours in missing_contours:
#     h,w =proj_masks[0].shape[:2]
#     missing_mask =create_mask([contours],w,h)
#     for cont in contours:
#         M = cv2.moments(cont)
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])

#         # for mask in proj_masks:
#         # print(proj_masks[0].shape)
                                              
        
#         for ratio_scale in np.arange(1.0, 2.0, 0.1):
#             s_project_masks = [scale_mask(mask,[cX,cY],ratio_scale) for mask in proj_masks]

#             coincident_points =check_coincident_point(real_labels,real_masks,project_labels,s_project_masks,missing_mask)
#             print(ratio_scale,coincident_points)

#         img= cv2.imread('/home/datpv/Downloads/missing/totaled.jpg')
#         img = scale_mask(img,[cX,cY],1.3)

#         for new_cont in missing_contours:
#             cv2.drawContours(img, new_cont, -1, (255, 0, 0), 3)
#         cv2.imshow('kkkkkkk',img)
#         cv2.waitKey(0)
#         s_project_masks = [scale_mask(mask,[cX,cY],1.3) for mask in proj_masks]

#         for label, mask in zip(project_labels,s_project_masks):
#             if cv2.countNonZero(mask)==0: 
#                 continue
#             overlap_area = cv2.bitwise_and(missing_mask, mask)
#             cv2.imshow('kkkkkkk',missing_mask*255)
#             cv2.waitKey(0)
#             print(label,cv2.countNonZero(overlap_area) /cv2.countNonZero(mask) )                                                                                                                                                