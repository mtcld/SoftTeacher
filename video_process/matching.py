import sys
sys.path.append('/datadriver/dat/videos/SuperGluePretrainedNetwork')

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import time 
import cv2

from video_process.SuperGluePretrainedNetwork.models_matching.matching import Matching
from video_process.SuperGluePretrainedNetwork.models_matching.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


device = 'cuda:0'
print('Running inference on device \"{}\"'.format(device))
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


# img1_path='assets/freiburg_sequence/1341847980.722988.png'

# img2_path ='assets/freiburg_sequence/1341847983.738736.png'
# img1=cv2.imread(img1_path)
# img2=cv2.imread(img2_path)
matching = Matching(config).eval().to(device)

def estimate_position(img1,img2,roi):
    # Load the image pair.
    image0, inp0, scales0 = read_image(
        img1, device, [1072, 1072], 0, False)
    image1, inp1, scales1 = read_image(
        img2, device, [1072, 1072], 0, False)



    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    if len(mconf) <250 :
        return None
    M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC,5.0)
    h,w = image0.shape
    pts = np.float32(roi).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    return dst
