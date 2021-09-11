import os
import torch
import numpy as np
import glob
import cv2
import copy
from random import shuffle
import random
from torch.utils import data
from .init_val import *


class airsim_fsl_dataset(data.Dataset):

    def __init__(
            self,
            data,
            split="train",
            subsplit=None,
            is_transform=False,
            img_size=(512, 512),
            augmentations=None,
            img_norm=True,
            version="airsim",
    ):

        #########################
        # dataloader parameters
        #########################
        self.name2color = name2color
        self.name2id = name2id
        self.id2name = id2name
        self.splits = ['train', 'val', 'test']
        self.image_modes = ['scene', 'segmentation_decoded']
        self.weathers = ['async_rotate_fog_000_clear']
        self.all_edges = all_edges
        self.split_subdirs = {}
        self.ignore_index = 0
        self.mean_rgb = mean_rgb

        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 11
        self.img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))
        self.mean = np.array(self.mean_rgb[version])
        self.label = list(range(self.n_classes))
        self.stat = data[0]
        self.data = data[1]

        # name2id = {"person": 1,
        #            "sidewalk": 2,
        #            "road": 3,
        #            "sky": 4,
        #            "pole": 5,
        #            "building": 6,
        #            "car": 7,
        #            "bus": 8,
        #            "truck": 9,
        #            "vegetation": 10}

        self.split_label_dict = split_label_dict
        self.split_label = self.split_label_dict[split]
        print(self.stat, self.data.keys(), self.split_label)

        all_labels = []
        all_data = []
        self.accumulate = []
        cnt = 0
        for k in sorted(list(self.stat.keys())):
            # print(k, self.stat[k])
            all_labels.extend([k] * self.stat[k])
            all_data.extend(self.data[k])
            cnt += self.stat[k]
            self.accumulate.append(cnt)

        # print(all_labels)
        self.all_labels = all_labels
        self.all_data = all_data
        self.mode = 'norm'

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        # for gaining fixed validation sets
        if self.mode == 'dummy':
            return 1

        # first, get the 'label' of interest of this image
        # using accumulate bin

        idd = None

        for ix, acc_idx in enumerate(self.accumulate):
            if index < acc_idx:
                idd = ix
                break

        # the true class is mapping cls to the split categories
        cls = self.split_label[idd]

        # print(cls)
        # assert cls and cls > 0
        img_path, mask_path = self.all_data[index]
        # if not os.path.isfile(img_path):
        #     print(img_path)
        img, mask = np.array(cv2.imread(img_path), dtype=np.uint8)[:, :, :3], \
                    np.array(cv2.imread(mask_path), dtype=np.uint8)[:, :, 0]

        # this is for classification
        # i.e., only fg is retained
        # no background is retained
        roi_pos = (mask == cls)
        # roi_pos_inv = (mask != cls)

        # this is for segmentation
        # i.e., ONLY ONE fg is retained  (e.g., person)
        # bg is also retained
        # roi_pos_fg = (mask == cls) | (mask == 0)
        # roi_pos_fg_inv = (mask != cls) & (mask != 0)

        # this is for pre-train
        # all training categories are available
        roi_pos_fg = (mask == 0)
        for c1 in self.split_label:
            # print(c1)
            roi_pos_fg |= (mask == c1)

        ###############
        # save image
        ###############
        # img0 = np.zeros_like(img)
        # img0[roi_pos] = img[roi_pos]
        # tt = np.random.randint(100, size=(1, 1))[0]
        # cv2.imwrite('./tmp/%d_%d_0.png' % (tt, cls), img0)
        # img0 = np.zeros_like(img)
        # img0[roi_pos_fg] = img[roi_pos_fg]
        # cv2.imwrite('./tmp/%d_%d_1.png' % (tt, cls), img0)

        # mask is shape [512, 512]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = mask

        img, lbl = self.transform(img, lbl)
        img_roi, lbl_roi = self.hl_roi(img, lbl, roi_pos, None)
        img_roi_fg, lbl_roi_fg = self.hl_roi_fg(img, lbl, roi_pos_fg, None)

        # img = torch.from_numpy(img).float()
        # lbl = torch.from_numpy(lbl).long()

        img_roi = torch.from_numpy(img_roi).float()
        lbl_roi = torch.from_numpy(lbl_roi).long()

        img_roi_fg = torch.from_numpy(img_roi_fg).float()
        lbl_roi_fg = torch.from_numpy(lbl_roi_fg).long()

        return img_roi_fg, lbl_roi_fg, img_roi, lbl_roi, cls

    def transform(self, img, lbl):

        """transform
        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        # classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = lbl.astype(int)
        return img, lbl

    def hl_roi(self, img, lbl, roi_pos, roi_pos_inv):

        """transform
        :param img:
        :param lbl:
        """
        # print(img.shape)
        img_roi = np.zeros_like(img)
        img_roi[:, roi_pos] = img[:, roi_pos]

        lbl_roi = np.zeros_like(lbl)
        lbl_roi[roi_pos] = 1

        # classes = np.unique(lbl_roi)
        # print(classes)

        img_roi = img_roi.astype(float)
        lbl_roi = lbl_roi.astype(int)

        return img_roi, lbl_roi

    def hl_roi_fg(self, img, lbl, roi_pos, roi_pos_inv):

        """transform
        :param img:
        :param lbl:
        """
        # print(img.shape)
        img_roi = np.zeros_like(img)
        img_roi[:, roi_pos] = img[:, roi_pos]

        lbl_roi = np.zeros_like(lbl)
        lbl_roi[roi_pos] = lbl[roi_pos]

        # classes = np.unique(lbl_roi)
        # print(classes)

        img_roi = img_roi.astype(float)
        lbl_roi = lbl_roi.astype(int)

        return img_roi, lbl_roi


class airsim_fsl_preprocess_dataset(data.Dataset):

    def __init__(
            self,
            root,
            split="train",
            subsplit=None,
            is_transform=False,
            img_size=(512, 512),
            augmentations=None,
            img_norm=True,
            commun_label='None',
            version="airsim",
            target_view="target"

    ):

        #########################
        # dataloader parameters
        #########################
        self.name2color = name2color
        self.name2id = name2id
        self.id2name = id2name
        self.splits = ['train', 'val', 'test']
        self.image_modes = ['scene', 'segmentation_decoded']
        self.weathers = ['async_rotate_fog_000_clear']
        self.all_edges = all_edges
        self.split_subdirs = {}
        self.ignore_index = 0
        self.mean_rgb = mean_rgb

        # divide data to regions by distance, and split
        self.dataset_div = self.divide_region_n_train_val_test()
        # generate folder names
        self.split_subdirs = self.generate_image_path(self.dataset_div)

        self.commun_label = commun_label
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 11
        self.img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))
        self.mean = np.array(self.mean_rgb[version])
        self.label = list(range(self.n_classes))

        # name2id = {"person": 1,
        #            "sidewalk": 2,
        #            "road": 3,
        #            "sky": 4,
        #            "pole": 5,
        #            "building": 6,
        #            "car": 7,
        #            "bus": 8,
        #            "truck": 9,
        #            "vegetation": 10}

        self.split_label_dict = split_label_dict
        self.split_label = self.split_label_dict[split]

        # print(self.root)  # dataset/airsim-mrms-noise-data
        # print(self.dataset_div)
        # print(self.split_subdirs)
        print(target_view)  ######## 6agent
        print(self.root)
        print(self.split)
        print(self.is_transform, self.augmentations, self.img_norm, self.n_classes, self.img_size, self.mean)

        # Set the target view; first element of list is target view
        self.cam_pos = self.get_cam_pos(target_view)

        # Pre-define the empty list for the images
        self.imgs = {s: {c: {image_mode: [] for image_mode in self.image_modes} for c in self.cam_pos} for s in
                     self.splits}
        self.com_label = {s: [] for s in self.splits}

        self.img_group_label_cnt = {}
        self.img_group_label_2_index = {}

        k = 0
        # print(self.splits)
        # print(self.cam_pos[0]) # agent1
        for split in self.splits:  # [train, val]
            for subdir in self.split_subdirs[split]:  # [trajectory ]

                file_list = sorted(glob.glob(
                    os.path.join(root, 'scene', 'async_rotate_fog_000_clear', subdir, self.cam_pos[0], '*.png'),
                    recursive=True))

                for file_path in file_list:
                    ext = file_path.replace(root + "/scene/", '')
                    file_name = ext.split("/")[-1]
                    path_dir = ext.split("/")[1]

                    # print(ext, file_name, path_dir)
                    # ext:: async_rotate_fog_000_clear/-337_-172__-221_-172/agent1/frame001727.png
                    # file_name:: frame001727.png
                    # path_dir:: -337_-172__-221_-172

                    # Check if an image file exists in all views and all modalities
                    # TODO: what is segmentation_decoded  -- GT mask with uint8 class label 0~10
                    # see __get_item__() function
                    # print(self.image_modes) ['scene', 'segmentation_decoded']
                    list_of_all_cams_n_modal = [os.path.exists(
                        os.path.join(root, modal, 'async_rotate_fog_000_clear', path_dir, cam, file_name)) for modal in
                        self.image_modes for cam in self.cam_pos]

                    # [True, True, True, True, True, True, True, True, True, True, True, True]
                    # print(list_of_all_cams_n_modal)

                    if all(list_of_all_cams_n_modal):
                        k += 1
                        # Add the file path to the self.imgs 
                        for comb_modal in self.image_modes:
                            for comb_cam in self.cam_pos:
                                file_path = os.path.join(root, comb_modal, 'async_rotate_fog_000_clear', path_dir,
                                                         comb_cam, file_name)
                                self.imgs[split][comb_cam][comb_modal].append(file_path)

        print("Found %d %s images" % (len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]]), self.split))

    # <---- Functions for conversion of paths ----> 
    def tuple_to_folder_name(self, path_tuple):
        start = path_tuple[1]
        end = path_tuple[2]
        path = str(start[0]) + '_' + str(-start[1]) + '__' + str(end[0]) + '_' + str(-end[1]) + '*'
        return path

    def generate_image_path(self, dataset_div):

        # Merge across regions
        train_path_list = []
        val_path_list = []
        test_path_list = []
        for region in ['skyscraper', 'suburban', 'shopping']:
            for train_one_path in dataset_div['train'][region][1]:
                train_path_list.append(self.tuple_to_folder_name(train_one_path))

            for val_one_path in dataset_div['val'][region][1]:
                val_path_list.append(self.tuple_to_folder_name(val_one_path))

            for test_one_path in dataset_div['test'][region][1]:
                test_path_list.append(self.tuple_to_folder_name(test_one_path))

        split_subdirs = {}
        split_subdirs['train'] = train_path_list
        split_subdirs['val'] = val_path_list
        split_subdirs['test'] = test_path_list

        return split_subdirs

    def divide_region_n_train_val_test(self):

        region_dict = {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]}
        test_ratio = 0.25
        val_ratio = 0.25

        dataset_div = {'train': {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]},
                       'val': {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]},
                       'test': {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]}}

        process_edges = []
        # label and compute distance
        for i, path in enumerate(self.all_edges):
            process_edges.append(label_region_n_compute_distance(i, path))

            # region = process_edges[i][4]
            # distance = process_edges[i][3]
            region_dict[process_edges[i][4]][1].append(process_edges[i])
            region_dict[process_edges[i][4]][0] = region_dict[process_edges[i][4]][0] + process_edges[i][3]

        # split data by distance ratio 2/4 for training, 1/4 for val, 1/4 for testing
        for region_type, distance_and_path_list in region_dict.items():
            total_distance = distance_and_path_list[0]
            test_distance = total_distance * test_ratio
            val_distance = total_distance * val_ratio

            path_list = distance_and_path_list[1]
            tem_list = copy.deepcopy(path_list)

            random.seed(2019)
            shuffle(tem_list)

            sum_distance = 0

            # Test Set
            while sum_distance < test_distance * 0.8:
                path = tem_list.pop()
                sum_distance += path[3]
                dataset_div['test'][region_type][0] = dataset_div['test'][region_type][0] + path[3]
                dataset_div['test'][region_type][1].append(path)

            # Val Set
            while sum_distance < (test_distance + val_distance) * 0.8:
                path = tem_list.pop()
                sum_distance += path[3]
                dataset_div['val'][region_type][0] = dataset_div['val'][region_type][0] + path[3]
                dataset_div['val'][region_type][1].append(path)

            # Train Set
            dataset_div['train'][region_type][0] = total_distance - sum_distance
            dataset_div['train'][region_type][1] = tem_list

        return dataset_div

    def get_cam_pos(self, target_view):
        if target_view == "DroneNP":
            cam_pos = ["DroneNN_main", "DroneNP_main", "DronePN_main", "DronePP_main", "DroneZZ_main"]
        else:
            raise ValueError('Invalid target_view %s' % (target_view,))
        return cam_pos

    def __len__(self):
        """__len__"""
        return len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]])

    # for pre-processing data to get image mask content
    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        total_pixel = 512 ** 2
        for k, camera in enumerate(self.cam_pos):

            img_path, mask_path = self.imgs[self.split][camera]['scene'][index], \
                                  self.imgs[self.split][camera]['segmentation_decoded'][index]

            # print(img_path, mask_path)
            img, mask = np.array(cv2.imread(img_path), dtype=np.uint8)[:, :, :3], np.array(cv2.imread(mask_path),
                                                                                           dtype=np.uint8)[:, :, 0]
            # print(mask)

            for i in self.split_label_dict[self.split]:
                cnt = np.sum(mask == i)
                rt = cnt * 1.0 / total_pixel
                if i == 0 or (i != 1 and rt >= 0.01) or (i == 1 and rt >= 0.005):
                    self.img_group_label_cnt[i] = self.img_group_label_cnt.get(i, 0) + 1

                    if i not in self.img_group_label_2_index:
                        self.img_group_label_2_index[i] = []

                    self.img_group_label_2_index[i].append((img_path, mask_path))

        return 1
