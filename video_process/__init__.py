from video_process.yolo_models import Yolo
import sys
sys.path.insert(0, '/SoftTeacher/thirdparty/ScaledYOLO')

def init_yolo_models():
    # crack_yolo= Yolo(file_name='crack.pth',confident=0.5,iou_thres=0.5,device='0',imgsz_list=[1024])
    scratch_yolo= Yolo(file_name='checkpoint/scratch.pth',confident=0.5,iou_thres=0.5,device='2',imgsz_list=[1536])
    dent_yolo= Yolo(file_name='checkpoint/dent.pth',confident=0.47,iou_thres=0.5,device='2',imgsz_list=[1536])
    # missing_yolo= Yolo(file_name='missing.pt',confident=0.5,iou_thres=0.5,device='0',imgsz_list=[1443])

    # return {'scratch':scratch_yolo}
    # return {'crack':crack_yolo}
    return {'scratch':scratch_yolo,'dent':dent_yolo}

yolo_models=init_yolo_models()
# yolo_models = 1