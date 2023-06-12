import os
import sys
import time
import json
from copy import copy
import numpy as np
import warnings
import torchvision
import tqdm
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import os
from utils.distributed import get_world_size, set_seed, synchronize, cleanup

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.custom_dataset_data_loader import CustomDatasetDataLoader_19_imat, sample_data
from neptune.new.types import File
import PIL.Image as Image
from options.base_options import parser_imat_19_labels
from utils.saving_utils import save_checkpoints, save_checkpoint_mgpu
from utils.saving_utils import load_checkpoint, load_checkpoint_mgpu
from U2net_for_deep_fashion.testing_utils import get_dict_labels_deepfshaion
from networks import U2NET
import neptune.new as neptune


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
def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

    # Saving options in yml file
    option_dict = vars(opt)
    with open(os.path.join(opt.save_dir, "training_options.json"), "w") as outfile:
        json.dump(option_dict, outfile)

    for key, value in option_dict.items():
        print(key, value)
def dice_coef(y_true, y_pred, label):
    y_true_f = y_true.astype('uint8').flatten()
    y_pred_f = y_pred.astype('uint8').flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    dice = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    iou = (intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)
    return dice, iou
def dice_coef_multilabel(y_true, y_pred, labels_to_check):
    dice= 0
    iou = 0

    for label in labels_to_check:
        dice_for_label,\
        iou_for_label\
            = dice_coef(y_true==label, y_pred==label, label)

        info_labels_dict_dice[str(label)].append(dice_for_label)
        info_labels_dict_iou[str(label)].append(iou_for_label)
        dice += dice_for_label
        iou += iou_for_label
    return dice/len(labels_to_check), iou/len(labels_to_check),  # taking average
def compute_dice_score(pred, labels):
    dice = 0
    iou = 0

    for batch_index in range(pred.shape[0]):
        d0 = pred[batch_index]
        l = labels[batch_index]
        output_tensor = F.log_softmax(d0, dim=0)
        output_tensor = torch.max(output_tensor, dim=0, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

        label_arr = torch.squeeze(l, dim=0).cpu().numpy()
        labels_we_have = np.unique(label_arr).tolist()
        dice_im, iou_im = dice_coef_multilabel(label_arr, output_arr, labels_we_have)

        dice += dice_im
        iou += iou_im

    return dice/pred.shape[0], iou/pred.shape[0],
def get_dict_labels_deepfshaion():
    path = 'deep_fashion/new_Imat_labels'
    with open(path, 'r') as f:
        data = f.read().split('\n')
    data = [_.split(' ') for _ in data]
    labels_decode = {kv[0]:kv[1] for kv in data}
    print('Labels are:')
    print(labels_decode)

    return labels_decode

def training_loop_mereged_labels_of_deep_fashion(rank, opt):
    local_rank = rank

    if opt.distributed:
        # local_rank = int(os.environ.get("LOCAL_RANK"))
        # Unique only on individual node.
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        local_rank = 0

    print(local_rank)
    if local_rank==0:
        global run
        run = neptune.init(
            project="gleb4848431/Human-parsing-u2net",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMWEzNTQ5ZC1kYTA1LTQ2MjEtOGJiNS05MzhiNDI0YWU1Y2IifQ==",
        )

    global decode_labels_dict
    decode_labels_dict = get_dict_labels_deepfshaion()

    TOTAL_LABELS = 19
    u_net = U2NET(in_ch=3, out_ch=TOTAL_LABELS)
    if opt.continue_train:
        u_net = load_checkpoint(u_net, opt.unet_checkpoint)
    u_net = u_net.to(device)
    u_net.train()
    if opt.distributed:
        u_net = nn.parallel.DistributedDataParallel(
            u_net,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        print("Going super fast with DistributedDataParallel")

    # initialize optimizer
    optimizer = optim.Adam(
        u_net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2,cooldown=2, min_lr=0.00005, verbose=True)

    # train dataset
    custom_dataloader = CustomDatasetDataLoader_19_imat()
    custom_dataloader.initialize(opt)
    loader = custom_dataloader.get_loader()
    # val dataset
    val_loader = CustomDatasetDataLoader_19_imat()
    opt_val = copy(opt)
    opt_val.isTrain = False
    val_loader.initialize(opt_val)
    val_loader = val_loader.get_loader()
    if local_rank==0:
        dataset_size = len(custom_dataloader)
        print("Total number of images avaliable for training: %d" % dataset_size)
        print("Total number of images avaliable for validation: %d" % len(val_loader))
        # writer = SummaryWriter(opt.logs_dir)
        print("Entering training loop!")

    # loss function
    weights = np.array([1] * TOTAL_LABELS)
    # set some weights to 1.2 that are important upper, lower, foot
    set_this_to_val = np.array([1,2,4])
    mask = np.bincount(set_this_to_val, minlength=TOTAL_LABELS) > 0
    weights[mask] = 1.1

    weights = weights.astype(dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)

    palette = get_palette(TOTAL_LABELS)

    # Main training loop
    val_num = 0
    for e in range(opt.epoch):
        pbar = range(len(loader))
        if opt.distributed:
            loader.sampler.set_epoch(e)
        get_data = sample_data(loader)
        for itr in tqdm.tqdm(pbar):
            data_batch = next(get_data)
            image, label = data_batch
            image = Variable(image.to(device))
            label = label.type(torch.long)
            label = Variable(label.to(device))

            d0, d1, d2, d3, d4, d5, d6 = u_net(image)

            loss0 = loss_CE(d0, label)
            loss1 = loss_CE(d1, label)
            loss2 = loss_CE(d2, label)
            loss3 = loss_CE(d3, label)
            loss4 = loss_CE(d4, label)
            loss5 = loss_CE(d5, label)
            loss6 = loss_CE(d6, label)
            del d1, d2, d3, d4, d5, d6

            total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            for param in u_net.parameters():
                param.grad = None

            total_loss.backward()
            # if opt.clip_grad != 0:
            #     nn.utils.clip_grad_norm_(u_net.parameters(), opt.clip_grad)
            optimizer.step()

            if itr % opt.val_freq == 0 and itr!=0:
                val_num+= 1
                dice = 0
                iou = 0
                global info_labels_dict_dice
                info_labels_dict_dice = {k: [] for k, v in decode_labels_dict.items()}
                global info_labels_dict_iou
                info_labels_dict_iou = {k: [] for k, v in decode_labels_dict.items()}

                print('Entering vlidation loop!')
                bar = range(len(val_loader))
                # bar = range(20)
                data_val = sample_data(val_loader)


                with torch.no_grad():
                    sum_loss = 0.0

                    for itr_val in tqdm.tqdm(bar):
                        data_batch = next(data_val)
                        image, label = data_batch
                        image = image.to(device)
                        label = label.type(torch.long)
                        label = label.to(device)

                        d0_val, d1_val, d2_val, d3_val, d4_val, d5_val, d6_val = u_net(image)

                        loss0_val = loss_CE(d0_val, label)
                        loss1_val = loss_CE(d1_val, label)
                        loss2_val = loss_CE(d2_val, label)
                        loss3_val = loss_CE(d3_val, label)
                        loss4_val = loss_CE(d4_val, label)
                        loss5_val = loss_CE(d5_val, label)
                        loss6_val = loss_CE(d6_val, label)
                        del d1_val, d2_val, d3_val, d4_val, d5_val, d6_val

                        total_loss_val = loss0_val * 1.5 + loss1_val + loss2_val + loss3_val + loss4_val + loss5_val + loss6_val

                        if local_rank == 0:
                            run["val/total_loss"].log(total_loss_val)
                            run["val/loss0"].log(loss0_val)
                        sum_loss += loss0_val
                        # also adds info about every label in 'info_labels_dict'
                        dice_batch, iou_batch = compute_dice_score(d0_val, label)
                        dice+=dice_batch
                        iou+=iou_batch
                        if local_rank == 0:
                            # run["val/dice_batch"].log(dice_batch)
                            run["val/iou_batch"].log(iou_batch)
                    mean_loss = sum_loss/(len(val_loader))
                    scheduler.step(mean_loss)


                    iou /=(itr_val+1)

                    if local_rank == 0:
                        run["val/iou_mean"].log(iou)
                        run["val/mean_loss"].log(mean_loss)
                        run['lr'].log(scheduler.optimizer.param_groups[0]['lr'])

                    for (label_num,iou_list), (_, label_name) in zip(info_labels_dict_iou.items(), decode_labels_dict.items()):
                        if len(iou_list) ==0:
                            continue
                        else:
                            iou_for_label_mean = np.round(sum(iou_list)/len(iou_list), decimals=4)
                        if local_rank == 0:
                            run[f"iou/{label_name}_{label_num}"].log(iou_for_label_mean, step=(val_num*10))

            if itr % opt.image_log_freq == 0:

                output_tensor = F.log_softmax(d0, dim=1)
                output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
                output_tensor = output_tensor[0]
                output_tensor = torch.squeeze(output_tensor, dim=0)
                output_arr = output_tensor.cpu().numpy()

                label_arr = label[0]
                label_arr = torch.squeeze(label_arr, dim=0)
                label_arr = label_arr.cpu().numpy()

                output_arr = np.concatenate([
                    output_arr,
                    label_arr
                ], axis=1)
                output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
                output_img.putpalette(palette)
                output_img.save(os.path.join(opt.save_dir, "images", f'{itr}' + ".png"))

                torchvision.utils.save_image(image[0], os.path.join(opt.save_dir, "images", f'{itr}' + ".jpg"))

                if local_rank == 0:
                    run["valid/out_result"].log(File(os.path.join(opt.save_dir, "images", f'{itr}' + ".png")))
                    run["valid/out_result"].log(File(os.path.join(opt.save_dir, "images", f'{itr}' + ".jpg")))
            if local_rank == 0:
                run["train/total_loss"].log(total_loss)
                run["train/loss0"].log(loss0)
            # writer.add_scalar("total_loss", total_loss, itr)
            # writer.add_scalar("loss0", loss0, itr)

            # if itr % opt.save_freq == 0 and itr>0 and local_rank == 0:
            #     save_checkpoints(opt, val_num*10, u_net, loss0)
        itr += 1
        if local_rank == 0 and opt.distributed:
            save_checkpoint_mgpu(opt, itr, u_net, loss0)

        elif local_rank == 0:
            save_checkpoints(opt, itr, u_net, loss0)

    if int(local_rank) == 0:
        run.stop()

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, opt):
    ddp_setup(rank, world_size)
    training_loop_mereged_labels_of_deep_fashion(rank, opt)
    destroy_process_group()

if __name__ == "__main__":
    opt = parser_imat_19_labels()
    options_printing_saving(opt)
    # training_loop_mereged_labels_of_deep_fashion(opt)
    import sys
    distributed = bool(sys.argv[1])
    # lr = float(sys.argv[2])
    opt.distributed = distributed

    try:
        if opt.distributed:
            print("Initialize Process Group...")
            # torch.distributed.init_process_group(backend="nccl", init_method="env://")
            # synchronize()
            set_seed(1000)
            world_size = torch.cuda.device_count()
            mp.spawn(main, args=(world_size, opt), nprocs=world_size)
        else:
            training_loop_mereged_labels_of_deep_fashion(0, opt)

        print("Exiting..............")
    except KeyboardInterrupt:
        cleanup(opt.distributed)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        cleanup(opt.distributed)