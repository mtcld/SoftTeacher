import numpy as np
import cv2

def get_largest_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_list = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours)==0:
        return None
    return contours[np.argmax(area_list)]
    
def get_largest_mask(mask):
    cnt=get_largest_contours(mask)
    if cnt is None:
        return mask
    mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cnt=np.array(cnt, dtype=int)
    cv2.fillPoly(mask, np.array([cnt]), 1)
    return mask

def get_mask_center(mask):
    max_contour=get_largest_contours(mask)

    if(max_contour is None or cv2.contourArea(max_contour) <10):
        return -1,-1

    M = cv2.moments(max_contour)

    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

def draw_contour(mask,thickness=13):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height,width=mask.shape[:2]
    blank_image = np.zeros(mask.shape[:2] + (3,), np.uint8)
    new_contours=[]
    for cont in contours:
        area= cv2.contourArea(cont)
        if area >0:
            new_contours.append(cont)
    blank_image=cv2.drawContours(blank_image, new_contours, -1, (1, 1, 1), thickness)
    imgray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imgray, 0.5, 1, 0)
    return new_contours,mask
def expend_mask(mask,thickness=13):
    contours,mask=draw_contour(mask,thickness)
    for cnt in contours:
        # cnt = scale_list_point(cnt)
        cv2.fillPoly(mask, np.array([cnt]), 1)
    return mask
    
def masks_edge(contours_list,masks):
    egde_mask = None
    if len(masks) ==0:
        return egde_mask
    blank_image = np.zeros(masks[0].shape[:2] + (3,), np.uint8)
    for i, contours in enumerate(contours_list):
        blank_image=cv2.drawContours(blank_image, contours, -1, (1, 1, 1), 15)
    
    imgray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    ret, egde_mask = cv2.threshold(imgray, 0.5, 1, 0)

    return  egde_mask

def get_contours(masks):
    contours_list=[]
    for mask in masks:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

    return contours_list


def check_overlap(mask_i,mask_j):
    intersection = np.logical_and(mask_i, mask_j)
    if (np.sum(intersection) == 0):
        return False
    return True

def get_neighbor(new_masks):
    number_masks=len(new_masks)
    matrix_neighbor = np.full((number_masks, number_masks), False, dtype=bool)
    for i in range(len(new_masks)-1):
        mask_i=new_masks[i]
        for j in range(i+1,len(new_masks),1):
            mask_j = new_masks[j]
            if check_overlap(mask_i,mask_j):
                matrix_neighbor[i][j]=True
                matrix_neighbor[j][i] = True
    return matrix_neighbor

def check_mask_inside_mask(mask_i,mask_j,thresh=0.3,check_max=True):
    if mask_i is None or mask_j is None:
        return False
    intersection = cv2.bitwise_and(mask_i, mask_j)
    area_intersection=cv2.countNonZero(intersection)
    area_mask_i=cv2.countNonZero(mask_i)
    area_mask_j = cv2.countNonZero(mask_j)

    if check_max:
        if area_intersection/max(area_mask_i,area_mask_j) > (1-thresh):
            return True
        return False
    else :
        if min(area_mask_i,area_mask_j)==0:
            return False 
        if area_intersection/min(area_mask_i,area_mask_j) > (1-thresh):
            return True
        return False

def add_mask2masks(t_mask,s_masks):
    if t_mask is None:
        return t_mask
    for s_mask in s_masks:
        t_mask=cv2.bitwise_or(t_mask, s_mask)
    return t_mask

def get_bbox(mask):
    # max_cnt= get_max_contours(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return [0,0,mask.shape[1],mask.shape[0]]

    l=mask.shape[1]
    t=mask.shape[0]
    r=0
    b=0
    for cont in contours:
        x, y, rw, rh = cv2.boundingRect(cont)
        l=min(l,x)
        t = min(t, y)
        r = max(r, x+rw)
        b = max(b, y+rh)
        
    bbox_car = [l, t, r-l, b-t]
    return bbox_car

def create_id_mask(contours, resize_dim):
	mask = np.zeros((int(resize_dim[0]), int(resize_dim[1])), dtype=np.uint8)
	for i, contour_sub in enumerate(contours):
		for cnt in contour_sub:
			cnt=np.array(cnt, dtype=int)
			cv2.fillPoly(mask, np.array([cnt]), i+1)
	return mask

def process_front_bumper_mask(labels,masks,font_bumper_mask,right_side=None):
    centers=[]
    for label, mask in zip(labels,masks): 
        if 'grille' in label or 'flp_front_license_plate' in label:
            centers.append(np.array(get_mask_center(mask)))
            mask=cv2.bitwise_not(mask)
            font_bumper_mask=cv2.bitwise_and(font_bumper_mask,mask)

    
    center_bumper=None
    if len(centers)>0:
        center_bumper=np.mean(np.array(centers),axis=0)
    
    if center_bumper is None:
        return font_bumper_mask,None,0

    center_bumper_x=int(center_bumper[0])
    
    or_h,or_w=font_bumper_mask.shape[:2]

    right_font_bumper=font_bumper_mask[0:or_h,0:center_bumper_x]
    
    right_font_bumper = cv2.copyMakeBorder(right_font_bumper, 0, 0, 0, or_w-center_bumper_x, cv2.BORDER_CONSTANT)
    
    left_font_bumper=font_bumper_mask[0:or_h,center_bumper_x:or_w-1]
    left_font_bumper = cv2.copyMakeBorder(left_font_bumper, 0, 0, center_bumper_x+1, 0, cv2.BORDER_CONSTANT)

    if right_side is not None:
        if right_side :
            return right_font_bumper,True,center_bumper_x
        else:
            return left_font_bumper,False,center_bumper_x
            
    if cv2.countNonZero(right_font_bumper) > cv2.countNonZero(left_font_bumper) :
        
        return right_font_bumper,True ,center_bumper_x
    return left_font_bumper,False ,center_bumper_x


def create_mask(contours,w,h):
    mask = np.zeros((h, w,1), dtype=np.uint8)
    for contour in contours:
        for cnt in contour:
            cnt = np.array(cnt, dtype=int)
            cv2.fillPoly(mask, np.array([cnt]),1)
    return mask

def convex_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    h,w= mask.shape[:2]
    return create_mask([hull_list],w,h)

def fill_hole_mask(mask):
    origin = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
    f_o= origin.copy()
    h, w = origin.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(f_o, mask, (0,0), 255)
    f_o= cv2.bitwise_not(f_o)
    f_o=cv2.bitwise_or(origin,f_o)
    f_o=f_o[10:h-10,10:w-10]
    return f_o

def create_mask(contours,w,h):
    mask = np.zeros((h, w,1), dtype=np.uint8)
    for contour in contours:
        for cnt in contour:
            cnt = np.array(cnt, dtype=int)
            cv2.fillPoly(mask, np.array([cnt]),1)
    return mask