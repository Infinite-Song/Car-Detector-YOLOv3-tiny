# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import sys
import numpy as np
import os.path
import glob
from multiprocessing import Process
from multiprocessing import Pool
import multiprocessing
import time

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
        
# Load names of classes
classesFile = "src/coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "src/yolov3-tiny.cfg";
modelWeights = "src/yolov3-tiny_60000.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

def video2imgs(video_name, frames_dir):
    time.sleep(1)
    cap = cv.VideoCapture(video_name)
    i = 0
    while cv.waitKey(1) < 0:
        hasFrame,frame=cap.read()
        if not hasFrame:
            break
        cv.imwrite(frames_dir + '/core-{:02d}.jpg'.format(i),frame)
        i = i + 1

def imgs2video(imgs_dir, save_name ,cap):
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv.CAP_PROP_FPS)
    video_writer = cv.VideoWriter(save_name, cv.VideoWriter_fourcc('M','J','P','G'), fps, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.jpg'))

    for i in range(len(imgs)):
        imgname = os.path.join(imgs_dir, 'core-{:02d}.jpg'.format(i))
        frame = cv.imread(imgname)
        video_writer.write(frame.astype(np.uint8))

    video_writer.release()

def display(src):
    # display video
    wnd = 'OpenCV Video'
    #获得视频的格式
    cap = cv.VideoCapture(src)
    #获得码率及尺寸
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    cv.namedWindow(wnd, flags=0)
    #读帧
    success, frame = cap.read()
    while success :
        cv.imshow(wnd, frame) #显示
        cv.waitKey(int(1000/fps)) #延迟
        success, frame = cap.read() #获取下一帧

def frame_process(i, frames_dir, output_frames_dir):
    imgname = os.path.join(frames_dir, 'core-{:02d}.jpg'.format(i))
    output_frame = os.path.join(output_frames_dir, 'core-{:02d}.jpg'.format(i))
    frame = cv.imread(imgname)
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    cv.imwrite(output_frame, frame.astype(np.uint8));

def detect(video):
    time.sleep(1)
    if (video):
        # Open the video file
        if not os.path.isfile(video):
            sys.exit(1)
        if not video.split('.')[-1]=='avi':
            sys.exit(1)
        
        cap = cv.VideoCapture(video)
        videoname=video.split('/')[-1]
        
        frames_dir = videoname.split('.')[0]
        if not os.path.exists(frames_dir):
            os.mkdir(frames_dir)
        output_frames_dir = './output_frames'
        if not os.path.exists(output_frames_dir):
            os.mkdir(output_frames_dir)
        output_file = 'out.avi'
#        thread1=threading.Thread(target=video2imgs(video, frames_dir),name='video2imgs')
        imgs = glob.glob(os.path.join(frames_dir, '*.jpg'))
        
        '''
        # using for loop
        for i in range(len(imgs)):
            frame_process(i, frames_dir, output_frames_dir)
        '''
        # using multiprocessing
        pool = Pool(processes=2)
        for x in range(2055):
            pool.apply_async(frame_process, args=(x,frames_dir, output_frames_dir))
        pool.close()
        pool.join()
        
        
        imgs2video(output_frames_dir, output_file, cap)

