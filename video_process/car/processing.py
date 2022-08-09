import numpy as np
import cv2

class CarProcessing():
    def __init__(self):
        pass
    
    def process(self,pred_json):
        car_pred = pred_json['car']
        car_masks = [mask for i, mask in enumerate(car_pred['masks']) if (car_pred['labels'][i] == 'car' or car_pred['labels'][i] == 'truck')]
        largest_mask,all_mask = self.get_largest_car_mask(car_masks)
        return largest_mask,all_mask

    def get_largest_car_mask(self,masks): 
        if (len(masks) == 0):
            return None,None
        all_mask=np.zeros_like(masks[0],np.uint8) 
        largest_mask = np.ones_like(masks[0],np.uint8)
        img_area=cv2.countNonZero(largest_mask)
        max_area=0
        for i, mask in enumerate(masks):
            mask = mask.astype(np.uint8)
            all_mask= cv2.bitwise_or(all_mask,mask)
            area=cv2.countNonZero(mask)
            if area/(img_area) <0.1:
                iv=np.bitwise_not(mask)
                largest_mask=cv2.bitwise_and(iv,largest_mask)
                continue 
            if area > max_area:
                largest_mask= mask
                max_area=area

        return largest_mask,all_mask
