import argparse
import os

import PIL.Image
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.nn as nn

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu, load_checkpoint

from networks import U2NET
from networks.u2net_double import U2NET_double
from U2net_for_deep_fashion.constants import ORIG_MEAN, ORIG_WIDTH, ORIG_STD, ORIG_HEIGHT
import albumentations as A

def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette
def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PosEstimation")

    parser.add_argument("--input", type=str, default='')
    parser.add_argument("--output", type=str, default='')
    parser.add_argument("--in_root", type=str, default='')
    parser.add_argument("--in_labels", type=str, default='')
    parser.add_argument("--out_root", type=str, default='')
    parser.add_argument("--single_folder", type=str, default='False', choices=['True', 'False', 'tf', 'double'])
    parser.add_argument("--folder_name", type=str, default='image')
    parser.add_argument("--loss_count", type=str,default='False', choices=['True', 'False'])


    return parser.parse_args()
def prcess_dataet(root, out_root, folder_name):
    for cat in os.listdir(root):
        folder = os.path.join(root, cat, folder_name)
        new_folder = os.path.join(out_root, cat, folder_name)
        try:
            os.makedirs(new_folder)
        except:
            pass
        run_inference_trained_check(folder, new_folder)

def run_inference_trained_check(image_dir, result_dir, check_path='results/Imat_and_DEEP/checkpoints/itr_00004036_u2net_0.04824_BEST.pth'):
# def run_inference_trained_check(image_dir, result_dir, check_path='results/Imat_19_aug_fixed/checkpoints/itr_00002830_u2net_0.03283.pth'):
    # def run_inference_trained_check(image_dir, result_dir, check_path='results/imat_19_fixed/checkpoints/itr_00000480_u2net_0.09725.pth'):
    # def run_inference_trained_check(image_dir, result_dir, check_path='results/imat_19/checkpoints/itr_00000060_u2net_0.08547_MAIN.pth'):

    os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = check_path
    device = "cuda"
    do_palette = True
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [transforms.Normalize(ORIG_MEAN, ORIG_STD)]
    trans_resize = A.Compose([
        A.Resize(height=512, width=512, interpolation=cv2.INTER_NEAREST)
    ])
    TOTAL_CLASSES = 19
    TOTAL_CLASSES_FINAL = 25



    transform_rgb = transforms.Compose(transforms_list)
    net = U2NET(in_ch=3, out_ch=TOTAL_CLASSES)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    palette = get_palette(TOTAL_CLASSES_FINAL)

    images_list = sorted(os.listdir(image_dir))
    pbar = tqdm(total=len(images_list))
    for image_name in images_list:
        try:
            img = Image.open(os.path.join(image_dir, image_name))
        except:
            continue
        im = np.array(img)
        shape = im.shape
        if len(shape)!=3:
            print(f'GRAY, shape[2] {shape}: ',image_name)
            continue
        img = trans_resize(image=im)['image']
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)



        output_tensor = net(image_tensor.to(device))

        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy().astype("uint8")
        trans_resize_back = A.Compose([
            A.Resize(height=shape[0], width=shape[1], interpolation=cv2.INTER_NEAREST)
        ])
        output_arr = trans_resize_back(image=output_arr)['image']
        output_arr = relabel_to_our_labels(output_arr)
        output_img = Image.fromarray(output_arr, mode="L")



        if do_palette:
            output_img.putpalette(palette)
        output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))

        pbar.update(1)

    pbar.close()
def relabel_to_our_labels(mask):
    relabel_arr = [False, 6, 7, False, False, 1, 5, 2, 9, 3, 13, False, 4, 14, 16, 15, 18, 17, 12, 11, 10, False, 8]
    new_mask = np.zeros(mask.shape)
    uniq = set(np.unique(mask).tolist())
    for old, new in enumerate(relabel_arr):
        if not new:
            continue
        if new not in uniq:
            continue
        new_mask[mask==new] = old
    return new_mask.astype('uint8')
def run_inference_trained_check_calac_loss(image_dir, result_dir, in_labels, check_path='results/Imat_19_aug_fixed/checkpoints/itr_00002830_u2net_0.03283.pth'):
    # def run_inference_trained_check(image_dir, result_dir, check_path='results/imat_19_fixed/checkpoints/itr_00000480_u2net_0.09725.pth'):
    # def run_inference_trained_check(image_dir, result_dir, check_path='results/imat_19/checkpoints/itr_00000060_u2net_0.08547_MAIN.pth'):

    os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = check_path
    device = "cuda"
    do_palette = True
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [transforms.Normalize(ORIG_MEAN, ORIG_MEAN)]
    trans_resize = A.Compose([
        A.Resize(height=512, width=512, interpolation=cv2.INTER_NEAREST)
    ])
    TOTAL_CLASSES = 19

    # loss function
    weights = np.array([1] * TOTAL_CLASSES)
    # set some weights to 1.2 that are important upper, lower, foot
    set_this_to_val = np.array([1,2,3,4,5,15,16,17,18])
    mask = np.bincount(set_this_to_val, minlength=TOTAL_CLASSES) > 0
    weights[mask] = 1.5

    weights = weights.astype(dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)


    transform_rgb = transforms.Compose(transforms_list)
    net = U2NET(in_ch=3, out_ch=TOTAL_CLASSES)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    palette = get_palette(TOTAL_CLASSES)

    images_list = sorted(os.listdir(image_dir))
    pbar = tqdm(total=len(images_list))
    loss_list = []
    for image_name in images_list:
        img = Image.open(os.path.join(image_dir, image_name))
        mask = Image.open(os.path.join(in_labels, image_name.replace('.jpg','.png')))
        label = torch.as_tensor(np.array(mask), dtype=torch.int64)
        label = torch.unsqueeze(label, 0)

        img = np.array(img)
        # shape = img.shape
        # img = trans_resize(image=img)['image']
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        with torch.no_grad():
            d0_val, d1_val, d2_val, d3_val, d4_val, d5_val, d6_val = net(image_tensor.to(device))

            loss0_val = loss_CE(d0_val, label.to(device))
            loss_list.append(
                (image_name.split('.')[0], loss0_val.cpu().numpy().tolist())
            )
            output_tensor = F.log_softmax(d0_val, dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_arr = output_tensor.cpu().numpy().astype("uint8")
            # trans_resize_back = A.Compose([
            #     A.Resize(height=shape[0], width=shape[1], interpolation=cv2.INTER_NEAREST)
            # ])
            # output_arr = trans_resize_back(image=output_arr)['image']
            output_img = Image.fromarray(output_arr, mode="L")



            if do_palette:
                output_img.putpalette(palette)
            output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))

            pbar.update(1)
    with open(os.path.join(result_dir, 'loss.txt'), 'w') as f:
        f.write(str(loss_list))
    pbar.close()
if __name__=='__main__':
    args = get_arguments()
    print(args)
    image_dir = args.input
    result_dir = args.output

    image_root = args.in_root
    result_root = args.out_root
    single_folder = args.single_folder
    folder_name = args.folder_name
    in_labels = args.in_labels

    loss_count = args.loss_count
    if loss_count == 'True':
        run_inference_trained_check_calac_loss(image_dir, result_dir,in_labels )
    elif image_root:
        prcess_dataet(image_root, result_root, folder_name)
    else:
        run_inference_trained_check(image_dir, result_dir)
    # /home/armg0/U2net_trainig/neural-networks/new_labels/To_train/converted_to_continious_resized_val/image-parse-v3
    # if single_folder=='True':
    #     run_inference(image_dir, result_dir)
    # elif single_folder=='tf':
    #     run_inference_trained_check(image_dir, result_dir)
    # elif single_folder=='double':
    #     run_inference_trained_check_Double(image_dir, result_dir)
    # else:

        # python3 infer.py --input /home/armg0/U2net_trainig/test_infer/in/100_sample/image --output /home/armg0/U2net_trainig/test_infer/out/100_sample_imat/image --single_folder  --folder_name image
        # python3 infer.py --in_root /home/armg0/U2net_trainig/test_infer/in/100_sample/image --out_root /home/armg0/U2net_trainig/test_infer/out/100_sample_imat/image  --folder_name person