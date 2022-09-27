from turtle import shape
from video_process.SuperGluePretrainedNetwork.models_matching.matching import Matching
from video_process.SuperGluePretrainedNetwork.models_matching.utils import read_image

from sklearn.cluster import KMeans
import torch 
import cv2
import numpy as np

torch.set_grad_enabled(False)

device = 'cuda:0'

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)


class Cross_check_pair():
    def __init__(self,hungarian):
        self.hungarian = hungarian
    
    def setup(self,info1,info2):
        self.info1 = info1
        self.info2 = info2

        self.draw1 = info1['frame'].copy()
        self.draw2 = info2['frame'].copy()

        self.kp1,self.kp2= self.get_matching_key_points(self.info1['frame'],self.info2['frame'])
    
  
    def get_matching_key_points(self,img1,img2):
        image0, inp0, scales0 = read_image(img1, device, [1072, 1072], 0, False)
        image1, inp1, scales1 = read_image(img2, device, [1072, 1072], 0, False)

        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid1 = matches > -1
        valid2 = conf > 0.3
        valid = np.logical_and(valid1,valid2)

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC,5.0)
        # self.transform_matrix = M

        return mkpts0,mkpts1
    
    def filter_keypoints_by_carpart(self,mask1,mask2):
        # valid1 = [mask1[int(p[1]),int(p[0])] for p in self.kp1]
        # valid2 = [mask2[int(p[1]),int(p[0])] for p in self.kp2]


        valid1 = []
        valid2 = []
        for p in self.kp1:
            # print('coord in mask : ',p[1],p[0],mask1.shape)

            if int(p[1]) >= mask1.shape[0] or int(p[0]) >= mask1.shape[1] : 
                valid1.append(0)
                continue

            if mask1[int(p[1]),int(p[0])]:
                valid1.append(1)
            else:
                valid1.append(0)
        
        for p in self.kp2:
            # print('coord in mask : ',p[1],p[0],mask2.shape)

            if int(p[1]) >= mask2.shape[0] or int(p[0]) >= mask2.shape[1] : 
                valid2.append(0)
                continue

            if mask2[int(p[1]),int(p[0])]:
                valid2.append(1)
            else:
                valid2.append(0)

        if sum(valid1) < 100 or sum(valid2) < 100:
            valid = np.logical_or(valid1,valid2)
        else:
            valid = np.logical_and(valid1,valid2)

        return self.kp1[valid],self.kp2[valid]

    def make_coordinate(self,mask1,mask2,space=20):
        # print('debug mask shape : ',mask1.shape,mask2.shape)
        kp1,kp2 = self.filter_keypoints_by_carpart(mask1,mask2)

        if len(kp1) < 20:
            return kp1,kp2
        
        kmean = KMeans(n_clusters=space,max_iter=50).fit(kp1)
        centers = kmean.cluster_centers_.astype(np.int32)

        valid = self.hungarian.matching_points(centers,kp1)
        kp1 = kp1[valid]
        kp2 = kp2[valid]

        return np.array([int(p[0]) for p in kp1]), np.array([int(p[0]) for p in kp2])
    
    # def estimate_bbox1(self,roi):
    #     pts = np.float32(roi).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,self.transform_matrix)

    #     return dst.astype(np.int32)
    
    def estimate_bbox2(self,roi_list):
        trackers = cv2.legacy.MultiTracker_create()

        for roi in roi_list:
            x1, y1, x2, y2 = roi
            tracker = cv2.legacy.TrackerCSRT_create()
            trackers.add(tracker,self.info1['frame'],[x1,y1,x2-x1,y2-y1])
        
        _,bboxes = trackers.update(self.info2['frame'])

        out = []
        for b in bboxes:
            b = np.int32(b)
            out.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

        return out

    def cross_check(self,info1,info2):
        def get_bbox(result,damage_list):
            roi_list = []
            idx_list = []

            for idx,d in enumerate(damage_list):
                if d[0] != 'scratch' :
                    continue

                b = result['scratch']['bboxes'][d[2]]
                roi = b

                roi_list.append(roi)
                idx_list.append(idx)

            return roi_list,idx_list
        
        self.setup(info1,info2)

        for carpart_label1,damage_list1 in self.info1['damage_result'].items():
            print('check car part ',carpart_label1)
            scratch_list1,ind_list1 = get_bbox(self.info1['result'],damage_list1)

            if len(scratch_list1) == 0:
                continue

            es_scratch_list1 = self.estimate_bbox2(scratch_list1)

            for b in scratch_list1:
                cv2.rectangle(self.draw1,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
            
            for b in es_scratch_list1:
                cv2.rectangle(self.draw2,(b[0],b[1]),(b[2],b[3]),(255,0,255),2)


            carpart_mask1 = self.info1['result']['carpart']['masks'][damage_list1[ind_list1[0]][3]]

            for carpart_label2,damage_list2 in self.info2['damage_result'].items():
                if carpart_label2 != carpart_label1:
                    continue

                scratch_list2,ind_list2 = get_bbox(self.info2['result'],damage_list2)

                if len(scratch_list2) == 0:
                    continue

                carpart_mask2 = self.info2['result']['carpart']['masks'][damage_list2[ind_list2[0]][3]]
                coord_kp1,coord_kp2 = self.make_coordinate(carpart_mask1,carpart_mask2)
                
                # print('image 2 ')
                coord_list2 = []
                for b in scratch_list2:
                    center = [int((b[0]+b[2])/2),int((b[1]+b[3])/2)]
                    coord = (coord_kp2<center[0]).sum()
                    coord_list2.append(coord)

                coord_list1 = []
                for b in scratch_list1:
                    center = [int((b[0]+b[2])/2), int((b[1]+b[3])/2)]
                    coord = (coord_kp1<center[0]).sum()
                    coord_list1.append(coord)
            
                self.info1,self.info2 = self.hungarian.matching_damages(carpart_label1,es_scratch_list1,ind_list1,coord_list1,self.info1,scratch_list2,ind_list2,coord_list2,self.info2)

                for b in scratch_list2:
                    cv2.rectangle(self.draw2,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
            # print('*'*200)  
            # cv2.imwrite('demo2.jpg',self.draw2)
            # cv2.imwrite('demo1.jpg',self.draw1)
                    
        return self.info1, self.info2