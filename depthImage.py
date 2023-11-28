import sys
sys.path.append('core')
# xx
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2

def load_image(imfile):
    
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def disparity_to_depth(disparity, baseline, focal_length):
    # Convert disparity to depth using the formula: depth = baseline * focal_length / disparity
    depth = baseline * focal_length / (disparity + 1e-6)  # Adding a small value to avoid division by zero
    return depth

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    # xx
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    depth_directory = Path(args.depth_directory)
    depth_directory.mkdir(exist_ok=True)

    # Assuming you have baseline and focal_length values
    baseline = 0.5707  # Replace with the actual baseline value
    focal_length = 645.24  # Replace with the actual focal length value

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)

            depth = disparity_to_depth(disp.squeeze(), baseline, focal_length)  

            file_stem = imfile1.split('/')[-2]
            filename = os.path.join(output_directory, f"{file_stem}.png")
            plt.imsave(output_directory / f"{file_stem}.png", disp.squeeze(), cmap='jet')

            # Save the depth map
            depth_file = os.path.join(depth_directory, f"{file_stem}_depth.png")
            plt.imsave(depth_directory / f"{file_stem}.png", depth, cmap='jet')


            # Choose a specific image coordinate (x, y)
            # x_coordinate = 327  # Replace with the actual x-coordinate of the point
            # y_coordinate = 253  # Replace with the actual y-coordinate of the point

            # x2 = 552
            # y2 = 187

            # # Get the depth value at the specified coordinate
            # depth_at_coordinate = depth[y_coordinate, x_coordinate]
            # x = depth[y2,x2]

            # # Calculate the real-world distance from the camera to the specified coordinate
            # real_distance = depth_at_coordinate  # Assuming the depth values are in the same unit as the baseline

            # print(f"Real-world distance at coordinate ({x_coordinate}, {y_coordinate}): {real_distance} m")
            # print(f"Real-world distance at coordinate ({x2}, {y2}): {x} m")
            # disp = np.round(disp * 256).astype(np.uint16)
            # cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/kitti15/kitti15.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/Middlebury/trainingH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/Middlebury/trainingH/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/ETH3D/two_view_training/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/ETH3D/two_view_training/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument('--depth_directory', help="directory to depth save output", default="./depth-output/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
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

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
