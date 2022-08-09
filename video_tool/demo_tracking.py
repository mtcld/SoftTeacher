import cv2 
import time 
from yolo_model import yolo
from damage import TrackingOject
from matching import estimate_position
import numpy as np

scratch=yolo('/datadriver/dat/models/yolo/scratch.pth',['scratch'],0.5,0.5)
dent=yolo('/datadriver/dat/models/yolo/dent.pth',['dent'],0.35,0.5)

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('20220504_142734.mp4')

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1072,1072))

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


trackers = cv2.MultiTracker_create()

colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0)]

track_flag=False
# Read until video is completed

color_count=0
count_frame =0
pre_frame=None 
roi_list=[]
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  print(frame.shape)
  if ret == True:      
    cv2.imwrite('frame.jpg',frame)
    if track_flag==False:
        # print('Create New tracking')
        scratch_output=scratch.inference('frame.jpg')
        for i,(bbox1,cof,label)  in enumerate(zip(scratch_output['boxes'],scratch_output['scores'],scratch_output['labels'])):
            x1,y1,x2,y2=bbox1
            # cv2.rectangle(frame,(x1,y1),(x2,y2),colors[color_count],1)  
            # roi=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            # roi_list.append(roi)
            # pre_frame=frame.copy()
            # track_flag=True 

            tracker = cv2.TrackerMedianFlow_create() 
            trackers.add(tracker, frame,  (x1,y1,x2-x1,y2-y1))
            track_flag=True 


    show=frame.copy()       

    if track_flag:

        # for mm in range(len(roi_list)-1,-1,-1):
        #     count_frame  +=1
        #     dst=estimate_position(pre_frame,frame,roi_list[mm])
        #     _,W_frame=frame.shape[:2]

        #     # if dst is None or np.min(dst) <0 or np.max(dst)> W_frame or count_frame>24:
        #     if dst is None or np.min(dst) <-30 or np.max(dst)> W_frame +30 or  count_frame>48:
        #         roi_list=[]
        #         color_count  +=1
        #         if color_count >9:
        #             color_count=0
        #         count_frame =0
        #         track_flag=False
        #         break  
          
        #     show = cv2.polylines(show,[np.int32(dst)],True,colors[color_count],3, cv2.LINE_AA)
        #     # update roi 
        #     roi_list[mm]=dst
        # pre_frame=frame.copy()        
        # roi=dst
        # pre_frame=frame.copy()
        

        count_frame  +=1
        if  count_frame >60:
          track_flag=False
          count_frame =0
          color_count  +=1
          continue
        success,boxes =trackers.update(frame)

        # if count_frame %5 ==0: 
        #   scratch_output=scratch.inference('frame.jpg')
        #   for bbox in scratch_output:

          
        
        if success:
            for bbox in boxes:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(show, (x, y), (x + w, y + h),colors[color_count], 2)
        else:
            color_count  +=1
            if color_count >=10:
                color_count=0
            track_flag=False
            trackers = cv2.MultiTracker_create()


    cv2.imwrite('show.jpg',show)
    # result.write(show)

    # # Display the resulting frame
    # cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    time.sleep(0.2)

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()