import cv2
import numpy as np
import sys
sys.path.append('core')
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from igev_stereo import IGEVStereo
import os
import argparse
from utils.utils import InputPadder
torch.backends.cudnn.benchmark = True
half_precision = True

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.backends.cudnn as cudnn
from numpy import random
from LaneDetect import LaneDetection

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *

from bridge_wrapper import YOLOv7_DeepSORT

 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True



parser = argparse.ArgumentParser(description='Iterative Geometry Encoding Volume for Stereo Matching and Multi-View Stereo (IGEV-Stereo)')
parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/kitti15/kitti15.pth')
parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="Left/*.png")
parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="Right/*.png")
parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

args = parser.parse_args()
model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
model.load_state_dict(torch.load(args.restore_ckpt))
model = model.module
model.to(DEVICE)
model.eval()


def find_3d_coordinates(u, v, Z, focal_length, width, height):

    u = u - int(width)//2
    v = -v + int(height)//2

    X = u * Z / focal_length
    Y = v * Z / focal_length
    return X, Y


def load_frame(frame):
    
    img = np.array(frame).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def disparity_to_depth(disparity, baseline, focal_length):
    # Convert disparity to depth using the formula: depth = baseline * focal_length / disparity
    depth = baseline * focal_length / (disparity + 1e-6)  # Adding a small value to avoid division by zero
    return depth


def calculateSpeed(Z, zFromPrevFrame, fps):
    dist = Z - zFromPrevFrame
    speed = (dist * fps) 
    speed = speed * 3.6
    return speed


if __name__ == '__main__':


    # Specify the paths to your two videos
    left_path = './data/1/Left.mp4'
    right_path = './data/1/Right.mp4'

    detector = Detector(classes = [0,2,3,4,6,8]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
    detector.load_model('./pretrained_models/Yolov7/yolov7.pt',) # pass the path to the trained weight file

    # Initialise  class that binds detector and tracker in one class
    yoloTracker = YOLOv7_DeepSORT(reID_model_path="./pretrained_models/deepsort/mars-small128.pb", detector=detector)

    # Open the video files
    left = cv2.VideoCapture(left_path)
    right = cv2.VideoCapture(right_path)

    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(left.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("./data/1/coordinates.avi", codec, fps, (width, height))
    print(height)
    print(width)

    # Assuming you have baseline and focal_length values
    baseline = 0.5707  # Replace with the actual baseline value
    focal_length = 645.24  # Replace with the actual focal length value

    frame_count = 0
    fps_list = np.array([])

    zFromPrevFrame = np.zeros(100)
    speed =  np.zeros(100)

    # videoWrite = cv2.VideoWriter('./Infe.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1242, 375))

    while True:
        # Read frames from the videos
        ret1, frame1 = left.read()
        ret2, frame2 = right.read()

        if not ret1 or not ret2:
            print('Video has ended or failed!')
            break

        yolo_dets = detector.detect(frame1.copy(), plot_bb = False)  # Get the detections
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        if yolo_dets is None:
            bboxes = []
            scores = []
            classes = []
            num_objects = 0
            
        else:
            bboxes = yolo_dets[:,:4]
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            scores = yolo_dets[:,4]
            classes = yolo_dets[:,-1]
            num_objects = bboxes.shape[0]

        
        names = []
        for i in range(num_objects): # loop through objects and use class index to get class name
            class_indx = int(classes[i])
            class_name = yoloTracker.class_names[class_indx]
            names.append(class_name)

        names = np.array(names)
        count = len(names)

        # Put num of object on screen
        cv2.putText(frame1, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)        


        features = yoloTracker.encoder(frame1, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

        cmap = plt.get_cmap('tab20b') #initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, yoloTracker.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        yoloTracker.tracker.predict()  # Call the tracker
        yoloTracker.tracker.update(detections) #  updtate using Kalman Gain

        if frame_count % 3 == 0:
            print("current:",frame_count)
            image1 = load_frame(frame1)
            image2 = load_frame(frame2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1_pad, image2_pad = padder.pad(image1, image2)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=half_precision):
                    disp = model(image1_pad, image2_pad, iters=16, test_mode=True)
                    disp = padder.unpad(disp)
            end.record()
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end)
            fps = 1000/runtime
            fps_list = np.append(fps_list, fps)
            if len(fps_list) > 3:
                fps_list = fps_list[-3:]
            avg_fps = np.mean(fps_list)
            print('Stereo runtime: {:.3f}'.format(1000/avg_fps))

            depth = disparity_to_depth(disp.squeeze(), baseline, focal_length)
        
        for track in yoloTracker.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
        
            color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
            color = [i * 255 for i in color]

            # Computing centre of bounding box
            if bbox[2] >= width:
                bbox[2] = width

            x_coordinate = (int(bbox[0]) + int(bbox[2]))//2
            y_coordinate = (int(bbox[1]) + int(bbox[3]))//2

            if x_coordinate > 1242:
                print(str(y_coordinate) + " " + str(x_coordinate))
            

            

            # if x_coordinate > width and y_coordinate > width:
            #     continue
                
            depth_at_coordinate = depth[y_coordinate, x_coordinate]

            depth_at_coordinate = depth_at_coordinate.cpu().numpy()
            Z = depth_at_coordinate

            
            X,Y = find_3d_coordinates(x_coordinate, y_coordinate, Z, focal_length, width, height)

            if frame_count % 3 == 0:
                speed[track.track_id] = calculateSpeed(Z, zFromPrevFrame[track.track_id], avg_fps )
                zFromPrevFrame[track.track_id] = Z



            cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.circle(frame1, (x_coordinate, y_coordinate),1, color, 2)
            cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame1, str(track.track_id) + " : " + str("{:.1f}".format(X)) + " " + str("{:.1f}".format(depth_at_coordinate)) ,(int(bbox[0]), int(bbox[1]-11)),0, 0.3, (255,255,255),1, lineType=cv2.LINE_AA)  
            # cv2.putText(frame1, str(track.track_id) + " : " + str("{:.1f}".format(speed[track.track_id]))  ,(int(bbox[0]), int(bbox[1]-11)),0, 0.3, (255,255,255),1, lineType=cv2.LINE_AA)  



        result = np.asarray(frame1)
        result = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        out.write(result)
        # Perform depth calculation every 3 frames
            # depth = (2*depth).data.cpu().numpy().squeeze().astype(np.uint8) 
            # depth = cv2.applyColorMap(depth, cv2.COLORMAP_PLASMA) 

            # Write the depth map visualization to the output video
            # cv2.waitKey(1)
            # videoWrite.write(depth)

            # videoWrite.write(depth)
        # Increment frame count
        frame_count += 1

    # Release the video capture objects
    # videoWrite.release()
    left.release()
    right.release()
    cv2.destroyAllWindows()