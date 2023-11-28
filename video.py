import cv2
import os

def create_video(image_folder, video_name='output_video.mp4', fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort the images by name
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Replace 'path/to/images/folder' with the actual path to your image folder
image_folder_path = './data/2/Right'

# Specify the output video name and frames per second (fps)
output_video_name = './data/2/Right.mp4'
fps = 10

create_video(image_folder_path, output_video_name, fps)
