import cv2
import numpy as np
import torch
import time
import os
import sys
import argparse
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import get_network
from data import get_loader
import torchvision.transforms as std_trnsf
from utils import joint_transforms as jnt_trnsf
from utils.metrics import MultiThresholdMeasures

def str2bool(s):
    return s.lower() in ('t', 'true', 1)

def has_img_ext(fname):
    ext = os.path.splitext(fname)[1]
    return ext in ('.jpg', '.jpeg', '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', help='path to ckpt file',type=str,
            default='./models/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth')
    parser.add_argument('--img_dir', help='path to image files', type=str, default='./data/Figaro1k')
    parser.add_argument('--networks', help='name of neural network', type=str, default='pspnet_resnet101')
    parser.add_argument('--save_dir', default='./overlay',
            help='path to save overlay images')
    parser.add_argument('--use_gpu', type=str2bool, default=True,
            help='True if using gpu during inference')

    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    img_dir = args.img_dir
    network = args.networks.lower()
    save_dir = args.save_dir
    device = 'cuda' if args.use_gpu else 'cpu'

    assert os.path.exists(ckpt_dir)
    assert os.path.exists(img_dir)
    assert os.path.exists(os.path.split(save_dir)[0])

    os.makedirs(save_dir, exist_ok=True)

    # prepare network with trained parameters
    net = get_network(network).to(device)
    state = torch.load(ckpt_dir)
    net.load_state_dict(state['weight'])


    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    durations = list()

    # prepare images
    img_paths = [os.path.join(img_dir, k) for k in sorted(os.listdir(img_dir)) if has_img_ext(k)]
    with torch.no_grad():
        for i, img_path in enumerate(img_paths, 1):
            print('[{:3d}/{:3d}] processing image... '.format(i, len(img_paths)))

            # Now, Input is not cropped one.
            # 1. Detect Box containing face (with some padding) -> using dlib.
            # 2. Make it square
            # 3. Save Cropped image (high-resolution)
            # 4. Down-sample to 250x250 and save it
            # 5. inference (segmentation) to make mask using down-sampled image.
            img = Image.open(img_path)
            
            img = np.array(img).astype(np.float32)

            print(img.shape)

            import dlib
            detector = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor('res/68.dat')
            boxes = detector(img)
            for face in faces: 
                # The face landmarks code begins from here 
                x1 = face.left() 
                y1 = face.top() 
                x2 = face.right() 
                y2 = face.bottom()
                size = max(x2-x1, y2-y1)

                x1 = max(0, int(x1-0.5*size))
                y1 = max(0, int(y1-0.5*size)) 

                square_size = min(2*size, img.shape[1] - y1, img.shape[2] - x1)

                cropped_img = img[x1: x1+square_size, y1: y1+square_size]
                cv2.imwrite("cropped.png", cropped_img) # TODO :: file path !!

            # resize (250 250) and save
            img = Image.fromarray(cropped_img, 'RGB')
            img = img.resize((250,250))
            img = np.array(img).astype(np.float32)
            cv2.imwrite("image250.png", img)

            print(img.shape)

            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)
            img = torch.from_numpy(img)
            data = torch.unsqueeze(img, dim=0)
            
            net.eval()
            data = data.to(device)

            # inference
            start = time.time()
            logit = net(data)
            duration = time.time() - start

            # prepare mask
            pred = torch.sigmoid(logit.cpu())[0].data.numpy() # 3 x 256 x 256, why cpu?
            mask = np.argmax(pred, axis=0) # 256 x 256, 0 or 1 or 2
            mh, mw = data.size(2), data.size(3)
            
            mask_copy = mask*127 # for grayscale image

            #path = os.path.join(save_dir, os.path.basename(img_path + "_mask") +'.png')

            cv2.imwrite("mask.png", mask_copy)

            break


    avg_fps = sum(durations)/len(durations)
    print('Avg-FPS:', avg_fps)