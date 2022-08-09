import os 
from thirdparty.ScaledYOLO.models.experimental import attempt_load
from thirdparty.ScaledYOLO.utils.datasets import LoadImagesBatch
from thirdparty.ScaledYOLO.utils.general import (check_img_size, non_max_suppression, scale_coords)
from thirdparty.ScaledYOLO.utils.torch_utils import select_device
# from car_damage.config.cfg import envs
import torch 
import numpy as np
from thirdparty.ScaledYOLO.utils.datasets import letterbox

class Yolo():
    def __init__(self,file_name='',confident=0.5,iou_thres=0.5,device='2',imgsz_list=[]):
        self.device = select_device(device)
        print('debug device :',self.device)
        self.half = self.device .type != 'cpu'
        self.conf_score=confident
        self.iou_thres=iou_thres

        self.model = attempt_load(file_name, map_location=self.device )  # load FP32 model
        if self.half:
            self.model.half()
        self.file_name=file_name
        self.imgsz_list=imgsz_list
    
    def pre_processing(self,image,image_size):
        h0,w0 = image.shape[:2]
        img,ratio,pad = letterbox(image, new_shape=image_size,auto=False)
        h,w = img.shape[:2]
        shape = (h0, w0), (((h-2*pad[1]) / h0, (w-2*pad[0]) / w0), pad)
        #print('after leter shape :',img.shape)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        #return path, img, image, self.cap
        return img, image, shape


    def __call__(self,image):
        # create dataset
        
        boxes = []
        confs=[]
        # shapes=[[0]]
        # for imgsz in self.imgsz_list:
        
        # dataset = LoadImagesBatch(img_path,img_size=imgsz)
        # load dataset and create batch with size equal number of given img_paths
        batch = []
        shapes = []
        paths = []
        # for path, img, im0s, shape in dataset:

        imgsz = check_img_size(self.imgsz_list[0], s=self.model.stride.max())  # check img_size
        img,img0,shape = self.pre_processing(image,imgsz)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        batch.append(img)
        shapes.append(shape)
        # paths.append(path)
        batch = torch.cat(batch)

        # run batch inference
        with torch.no_grad():
            pred = self.model(batch, augment=False)[0]

        # Apply MNS
        pred = non_max_suppression(pred, self.conf_score, self.iou_thres, agnostic=True)

        # Post process


        # img=cv2.imread(img_path)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # remove batch padding and rescale each image to its original size
                det[:, :4] = scale_coords(batch.shape[2:], det[:, :4], shapes[i][0],shapes[i][1]).round()
            
                # convert tensor to list and scalar
                
                for *xyxy, conf, cls in det:
                    rec = torch.tensor(xyxy).view(1,4).view(-1).int().tolist()
                    boxes.append(rec)
                    confs.append(conf)

        return boxes,confs,shapes[0][0]