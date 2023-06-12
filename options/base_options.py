import os.path as osp
import os
from U2net_for_deep_fashion.constants import *

# class parser(object):
#     def __init__(self):
#         self.name = "training_hum_pars_0"  # Expriment name
#         self.image_folders_root = "/media/globolik/SDD/projects_dataset/Upload/cloth-segmentation/deep_fashion/only_seg/cat"  # image folder path
#         self.image_folders_root_validation = "/media/globolik/SDD/projects_dataset/Upload/cloth-segmentation/deep_fashion/only_seg_resized_samall_res/cat_val"  # image folder path val
#         self.distributed = False  # True for multi gpu training
#         self.isTrain = True
#         self.normalization_like_in_article = True
#
#         # Mean std params
#         if self.normalization_like_in_article:
#             self.mean = ORIG_MEAN
#             self.std = ORIG_STD
#         else:
#             self.mean = MEAN
#             self.std = STD
#
#
#         self.batchSize = 1  # 12
#         self.nThreads = 4  # 3
#         self.max_dataset_size = float("inf")
#
#         self.serial_batches = False
#         self.continue_train = False
#         if self.continue_train:
#             self.unet_checkpoint = "U2net_for_deep_fashion/checkpoints/cloth_segm_unet_surgery.pth"
#
#         self.save_freq = 1000
#         self.val_freq = 51
#         self.image_log_freq = 500
#
#         self.iter = 100
#         self.lr = 0.0002
#         self.clip_grad = 5
#         self.epoch = 2
#         self.logs_dir = osp.join("logs", self.name)
#         self.save_dir = osp.join("results", self.name)
# class parser_original(object):
#     def __init__(self):
#         self.name = "orig_DF"  # Expriment name
#         self.image_folders_root = "deep_fashion/only_seg_resized/cat"  # image folder path
#         self.image_folders_root_validation = "deep_fashion/only_seg_resized/cat_val"  # image folder path val
#         self.distributed = False  # True for multi gpu training
#         self.isTrain = True
#         self.normalization_like_in_article = True
#
#         # Mean std params
#         if self.normalization_like_in_article:
#             self.mean = ORIG_MEAN
#             self.std = ORIG_STD
#         else:
#             self.mean = MEAN
#             self.std = STD
#         self.batchSize = 2  # 12
#         self.nThreads = 4  # 3
#         self.max_dataset_size = float("inf")
#
#         self.serial_batches = False
#         self.continue_train = True
#         if self.continue_train:
#             self.unet_checkpoint = "results/orig_DF/checkpoints/itr_00005904_u2net_0.09611.pth"
#
#         self.save_freq = 1000
#         self.val_freq = 1000
#         self.image_log_freq = 500
#
#         self.iter = 11000
#         self.lr = 0.0001
#         self.clip_grad = 5
#         self.epoch = 2
#         self.logs_dir = osp.join("logs", self.name)
#         self.save_dir = osp.join("results", self.name)
# class parser_train(object):
#     def __init__(self):
#         self.name = "training_hum_pars_0"  # Expriment name
#         self.image_folders_root = "new_labels/final_resized/cat_small_labels"  # image folder path
#         self.image_folders_root_validation = "new_labels/final_resized/cat_val"  # image folder path val
#         self.distributed = False  # True for multi gpu training
#         self.isTrain = True
#         self.mearged_dataset = True
#         self.normalization_like_in_article = True
#
#         # Mean std params
#         if self.normalization_like_in_article:
#             self.mean = ORIG_MEAN
#             self.std = ORIG_STD
#         else:
#             self.mean = MEAN
#             self.std = STD
#         self.freez_layesrs = False
#
#         self.batchSize = 2  # 12
#         self.nThreads = 4  # 3
#         self.max_dataset_size = float("inf")
#
#         self.serial_batches = False
#         self.continue_train = True
#         if self.continue_train:
#             self.unet_checkpoint = "results/training_hum_pars_0/checkpoints/itr_00000020_u2net_0.06477.pth"
#
#         self.save_freq = 1000
#         self.val_freq = 1000
#         self.image_log_freq = 500
#
#         self.iter = 11000
#         self.lr = 0.0001
#         self.clip_grad = 5
#         self.epoch = 2
#         self.logs_dir = osp.join("logs", self.name)
#         self.save_dir = osp.join("results", self.name)
# class parser_11_labels(object):
#     def __init__(self):
#         self.name = "11_labls"  # Expriment name
#         self.image_folders_root = "new_labels/11_l_resized/train"  # image folder path
#         self.image_folders_root_validation = "new_labels/11_l_resized/val"  # image folder path val
#         self.distributed = False  # True for multi gpu training
#         self.isTrain = True
#         self.mearged_dataset = True
#         self.normalization_like_in_article = False
#
#         # Mean std params
#         if self.normalization_like_in_article:
#             self.mean = ORIG_MEAN
#             self.std = ORIG_STD
#         else:
#             self.mean = MEAN
#             self.std = STD
#         self.freez_layesrs = False
#
#         self.batchSize = 2  # 12
#         self.nThreads = 4  # 3
#         self.max_dataset_size = float("inf")
#
#         self.serial_batches = False
#         self.continue_train = True
#         if self.continue_train:
#             # self.unet_checkpoint = "U2net_for_deep_fashion/checkpoints/cloth_segm_unet_surgery_11.pth"
#             self.unet_checkpoint = "results/11_labls/checkpoints/itr_00003657_u2net_0.12770.pth"
#
#         self.save_freq = 1000
#         self.val_freq = 1000
#         self.image_log_freq = 500
#
#         self.iter = 11000
#         self.lr = 0.00005
#         self.clip_grad = 5
#         self.epoch = 2
#         self.logs_dir = osp.join("logs", self.name)
#         self.save_dir = osp.join("results", self.name)
# class parser_9_labels(object):
#     def __init__(self):
#         self.name = "9_labls"  # Expriment name
#         self.image_folders_root = "new_labels/9_l/train"  # image folder path
#         self.image_folders_root_validation = "new_labels/9_l/val"  # image folder path val
#         self.distributed = False  # True for multi gpu training
#         self.isTrain = True
#         self.mearged_dataset = True
#
#         # Mean std params
#         self.mean = LABL_4_MEAN
#         self.std = LABL_4_STD
#
#
#         self.batchSize = 1  # 12
#         self.nThreads = 2  # 3
#         self.max_dataset_size = float("inf")
#
#         self.serial_batches = False
#         self.continue_train = True
#
#         self.save_freq = 3000
#         self.val_freq = 3000
#         self.image_log_freq = 1000
#
#         self.iter = 11000
#         self.lr = 0.00008
#         self.clip_grad = 5
#         self.epoch = 15
#         self.logs_dir = osp.join("logs", self.name)
#         self.save_dir = osp.join("results", self.name)
class parser_imat_19_labels(object):
    def __init__(self):
        self.name = "Imat_and_DEEP"  # Expriment name
        self.image_folders_root = "new_labels/DATSET_LAST/all"  # image folder path
        self.image_folders_root_validation = "new_labels/DATSET_LAST/all_val"  # image folder path val
        self.distributed = False  # True for multi gpu training
        self.isTrain = True

        # Mean std params
        self.mean = ORIG_MEAN
        self.std = ORIG_STD


        self.batchSize = 8  # 12
        self.nThreads = 4  # 3
        self.max_dataset_size = float("inf")

        self.shuffle = True
        self.continue_train = True
        # self.unet_checkpoint = "results/training_hum_pars_0/checkpoints/cloth_segm_unet_surgery_19.pth"
        # self.unet_checkpoint = "results/Imat_19_aug_fixed/checkpoints/itr_00002830_u2net_0.06251.pth"
        # self.unet_checkpoint = "results/Imat_19_aug_fixed/checkpoints/itr_00002830_u2net_0.09638.pth"
        self.unet_checkpoint = "results/Imat_and_DEEP/checkpoints/itr_00004036_u2net_0.04933_LAST_BIG_LR.pth"
        self.save_freq = 99999999
        self.val_freq = 4000
        self.image_log_freq = int(self.val_freq/2 - 1)

        self.lr = 0.0001
        self.clip_grad = 5
        self.epoch = 150
        self.logs_dir = osp.join("logs", self.name)
        self.save_dir = osp.join("results", self.name)
class parser_imat_19_labels_QAT(object):
    def __init__(self):
        self.name = "Imat_and_DEEP"  # Expriment name
        self.image_folders_root = "/media/globolik/SDD/pasing_data/DATSET_LAST/all_val"  # image folder path
        self.image_folders_root_validation = "/media/globolik/SDD/pasing_data/DATSET_LAST/all_val"  # image folder path val
        self.distributed = False  # True for multi gpu training
        self.isTrain = True

        # Mean std params
        self.mean = ORIG_MEAN
        self.std = ORIG_STD
        self.NUMBER_OF_ITEMS_TO_USE = 100

        self.batchSize = 2  # 12
        self.nThreads = 4  # 3
        self.max_dataset_size = float("inf")

        self.shuffle = True
        self.continue_train = False
        # self.unet_checkpoint = "results/training_hum_pars_0/checkpoints/cloth_segm_unet_surgery_19.pth"
        # self.unet_checkpoint = "results/Imat_19_aug_fixed/checkpoints/itr_00002830_u2net_0.06251.pth"
        # self.unet_checkpoint = "results/Imat_19_aug_fixed/checkpoints/itr_00002830_u2net_0.09638.pth"
        # self.unet_checkpoint = "results/Imat_and_DEEP/checkpoints/itr_00004036_u2net_0.04933_LAST_BIG_LR.pth"
        self.save_freq = 99999999
        self.val_freq = 50
        self.image_log_freq = int(self.val_freq/2 - 1)

        self.lr = 0.0001
        self.clip_grad = 5
        self.epoch = 150
        self.logs_dir = osp.join("/media/globolik/SDD/projects_dataset/Upload/cloth-segmentation/OPTIMIIZATION/save_model/u2net/log", self.name)
        self.save_dir = osp.join("/media/globolik/SDD/projects_dataset/Upload/cloth-segmentation/OPTIMIIZATION/save_model/u2net/save", self.name)
