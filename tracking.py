from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

detector = Detector(classes = [0,2,3,4,6,8]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('yolov7.pt',) # pass the path to the trained weight file

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

tracker.track_video("./Left.mp4", output="./runs/detect/exp/street.avi", show_live = False, skip_frames = 0, count_objects = True, verbose=1)