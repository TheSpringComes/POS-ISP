import cv2
import numpy as np
import torch
import random
import os
import hashlib
import math
import gzip
from torch.utils.data import Dataset
import torchvision
from imageio import imread
import sys
sys.path.append("yolov3")

from yolov3.utils.dataloaders import LoadImagesAndLabels
from yolov3.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                        letterbox, mixup, random_perspective)
from yolov3.utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                                  check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                                  xywh2xyxy, xywhn2xyxy, xyxy2xywhn)

from isp.unprocess_np import unprocess_wo_mosaic
from util import AsyncTaskManager

from multiprocessing.pool import Pool, ThreadPool
from tqdm import tqdm
from pathlib import Path
from yolov3.utils.dataloaders import img2label_paths, get_hash
import glob
from itertools import repeat
from config import cfg
HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', "npy"
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'

import os
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from path import Path
from skimage.transform import resize as imresize

class LoadImagesAndLabelsRAWReplaySeg(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False,
                 yaml_path='',
                 mode="train"):
        super(LoadImagesAndLabelsRAWReplaySeg, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.use_linear = use_linear

        data_dict = check_det_dataset(yaml_path)
        img_path = data_dict[mode]
        self.data_dict = data_dict

        cfg = DEFAULT_CFG
        cfg.mosaic = 0
        cfg.close_mosaic = 0
        cfg.degrees = 0
        cfg.translate = 0
        cfg.scale = 0
        cfg.shear = 0
        cfg.perspective = 0
        cfg.flipud = 0
        cfg.fliplr = 0
        cfg.hsv_h = 0
        cfg.hsv_s = 0
        cfg.hsv_v = 0
        cfg.task = "segment"
        dataset = build_yolo_dataset(cfg, img_path=img_path, batch=8, rect=False, data=data_dict, mode=mode)
        self.dataset = dataset

    def load_image(self, f, img_size):
        im = cv2.imread(f)
        assert im is not None, f'Image Not Found {f}'
        h0, w0 = im.shape[:2]
        r = img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        return im, (h0, w0), im.shape[:2]

    def __getitem__(self, index):

        return self.dataset[index]

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        shapes_list = []
        resized_shapes_list = []
        masks_list = []
        img_list = []
        cls_list = []
        bboxes_list = []
        batch_idx_list = []
        for i in range(len(batch)):
            res = self.__getitem__(batch[i])
            im_list.append(res['im_file'])
            shapes_list.append(res['ori_shape'])
            resized_shapes_list.append(res['resized_shape'])
            masks_list.append(res['masks'],)
            img_list.append(res['img'])
            cls_list.append(res['cls'])
            bboxes_list.append(res['bboxes'])
            batch_idx_list.append(res['batch_idx'])
        return im_list, shapes_list, resized_shapes_list, masks_list, img_list, cls_list, bboxes_list, batch_idx_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret

class LoadImagesAndLabelsRAW(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAW, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp

        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]
        img = img / 255.0

        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)

        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)

        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

class LoadImagesAndLabelsRAWV2(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAWV2, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp

        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]
        img = img / 255.0

        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)
        img = (img * 65535).astype(np.uint16)

        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)

        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        img = img.astype(np.float32) / 65535.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

class LoadImagesAndLabelsRAWHR(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAWHR, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp

        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]
        img = img / 255.0

        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)
        img_hr = img.copy()

        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)

        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        img_hr = img_hr.transpose((2, 0, 1))
        img_hr = np.ascontiguousarray(img_hr)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, torch.from_numpy(img_hr)

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes, img_hr = batch
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes, img_hr = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)

class LoadImagesAndLabelsRAWReplay(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAWReplay, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.use_linear = use_linear

    def __getitem__(self, index):

        hyp = self.hyp

        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]
        img = img / 255.0

        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)

        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)

        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)

        return im_list, label_list, path_list, shapes_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret

class LoadImagesAndLabelsNormalize(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        super(LoadImagesAndLabelsNormalize, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:

            img, labels = self.load_mosaic(index)
            shapes = None

            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:

            img, (h0, w0), (h, w) = self.load_image(index)

            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

def linearize_ProPhotoRGB(pp_rgb, reverse=False):
  if not reverse:
    gamma = 1.8
  else:
    gamma = 1.0 / 1.8
  pp_rgb = np.power(pp_rgb, gamma)
  return pp_rgb

def read_tiff16(fn):
  import tifffile
  import numpy as np
  img = tifffile.imread(fn)
  if img.dtype == np.uint8:
    depth = 8
  elif img.dtype == np.uint16:
    depth = 16
  else:
    print("Warning: unsupported data type {}. Assuming 16-bit.", img.dtype)
    depth = 16

  return (img * (1.0 / (2**depth - 1))).astype(np.float32)

def read_img(fn):
    if fn.endswith('.tif') or fn.endswith('.tiff'):
      image = read_tiff16(fn)
      image = linearize_ProPhotoRGB(image)
    else:
      image = cv2.imread(fn)[:, :, ::-1]
      if image.dtype == np.uint8:
        image = image / 255.0
      elif image.dtype == np.uint16:
        image = image / 65535.0
    return image

import os
import random
import torch
import torchvision
from torch.utils.data import Dataset

def crawl_folders(folders_list, dataset='kitti'):
    imgs = []
    depths = []
    type1 = None

    for folder in folders_list:
        if type1 is None:
            files = folder.files()
            for idx in range(len(files)):
                if files[idx][-4:] in [".jpg", ".png"]:
                    type1 = '*' + folder.files()[idx][-4:]
                    break

        current_imgs = sorted(folder.files(type1))

        for name, idx in zip(current_imgs, range(len(current_imgs))) :
            if 'temp' in name:
                current_imgs.pop(idx)

        if dataset == 'kitti':
            current_depth = sorted(folder.files('*.npy'))
        imgs.extend(current_imgs)
        depths.extend(current_depth)

    if len(imgs) != len(depths):
        print('Err in files. Are there .jpg_temp.png files?')
        breakpoint()
        raise NameError

    return imgs, depths

def raw4ch(img_raw):
    H_, W_, _ = img_raw.shape

    img_raw = np.sum(img_raw, axis=2)
    R = img_raw[1::2, 0::2]
    G1 = img_raw[0::2, 0::2]
    G2 = img_raw[1::2, 1::2]
    B = img_raw[0::2, 1::2]
    H_new = G2.shape[0]
    W_new = G2.shape[1]
    img_raw_ = np.zeros((H_new, W_new, 4))
    img_raw_[:,:,0] = R[:H_new, :W_new]
    img_raw_[:,:,1] = G1[:H_new, :W_new]
    img_raw_[:,:,2] = G2[:H_new, :W_new]
    img_raw_[:,:,3] = B[:H_new, :W_new]
    img_raw_   = np.clip(cv2.resize(img_raw_, (W_, H_), interpolation=cv2.INTER_CUBIC), 0, 255)
    return img_raw_

def bayer_to_bgr(inputs):
    B_, C_,H_,W_ = inputs.shape
    inputs_ = torch.FloatTensor(B_, 3, H_,W_, device=inputs.device)
    inputs_[:, 0,:,:] = inputs[:, 0,:,:]
    inputs_[:, 1,:,:] = (inputs[:, 1,:,:] + inputs[:, 2,:,:])/2
    inputs_[:, 2,:,:] = inputs[:, 3,:,:]
    return inputs_

def toTensor(images):
    tensors = []
    for im in images:

        im = np.transpose(im, (2, 0, 1))

        tensors.append((torch.from_numpy(im).float()/255).clamp(0.,1.))
        return torch.stack(tensors)

class KITTI(Dataset):
    def __init__(self, is_train=True, loadSize=(512, 512), gt_replace_rules=None, train_crop=True):
        super().__init__()
        self.is_train = is_train
        self.dataset = 'kitti'
        if self.is_train:

            self.root = Path(cfg.depth_train_dir)
            scene_list_path = self.root / 'train.txt'
            self.train_list = [self.root/folder[:-1] for folder in open(scene_list_path)]
            self.data, self.label = crawl_folders(self.train_list, self.dataset)
        else:
            self.root = Path(cfg.depth_test_dir)

            test_img_folder = os.path.join(self.root, 'color')
            test_label_folder = os.path.join(self.root, 'depth')
            self.data, self.label = crawl_folders([Path(test_img_folder), Path(test_label_folder)])

        self.num_images = len(self.data)
        self.indices = list(range(self.num_images))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_name = self.data[idx]
        gt_name  = self.label[idx]

        img = imread(img_name).astype(np.float32)
        img = raw4ch(img)

        if not self.is_train:
            img = imresize(img, (256,832))
        img_t = toTensor([img])
        bgr = bayer_to_bgr(img_t)

        img = bgr[0]

        gt = np.load(gt_name).astype(np.float32)
        if gt.ndim == 2:
            gt = gt[None, ...]
        elif gt.ndim == 3 and gt.shape[0] != 1:

            if gt.shape[-1] == 1:
                gt = np.transpose(gt, (2, 0, 1))
        gt = torch.from_numpy(gt)[0]
        return img, gt, img_name

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)

        im_list, gt_list, path_list = [], [], []
        for i in batch:
            im, gt, path = self.__getitem__(i)
            im_list.append(im)
            gt_list.append(gt)
            path_list.append(path)
        return im_list, gt_list, path_list

    def get_next_batch(self, batch_size):
        return self.get_next_batch_(batch_size)

class FiveKPair(Dataset):
    def __init__(self, path_img, loadSize, expert='expert'):
        super().__init__()
        self.path_img = path_img
        self.path_gt = path_img.replace('input', expert)
        print(expert)
        self.data_img = sorted(os.listdir(path_img))
        self.gt_img = sorted(os.listdir(self.path_gt))
        self.loadSize = loadSize
        self.resize = torchvision.transforms.Resize((512, 512))
        self.num_images = len(self.data_img)
        self.indices = list(range(self.num_images))
        self.is_test = ('test' in path_img) or ('val' in path_img)

    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, idx):
        name = str(os.path.join(self.path_img, self.data_img[idx]))
        gt_name = str(os.path.join(self.path_gt, self.data_img[idx]))
        img = read_tiff16(name)
        img = linearize_ProPhotoRGB(img)
        gt = read_tiff16(gt_name)
        img = torch.tensor(img).permute(2,0,1)
        gt = torch.tensor(gt).permute(2,0,1)

        if img.shape != gt.shape:
            breakpoint()
        C, H, W = gt.shape
        width = min(H, W)

        if not self.is_test:
            rnd_h = random.randint(0, max(0, H - width))
            rnd_w = random.randint(0, max(0, W - width))
        else:
            rnd_h = 0
            rnd_w = 0
        img = img[ :, rnd_h:rnd_h + width, rnd_w:rnd_w + width]
        gt = gt[ :, rnd_h:rnd_h + width, rnd_w:rnd_w + width]
        return self.resize(img), self.resize(gt), name

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        gt_list = []
        path_list = []
        for i in range(len(batch)):
            im, gt, path = self.__getitem__(batch[i])
            im_list.append(im)
            gt_list.append(gt)
            path_list.append(path)

        return im_list, gt_list, path_list

    def get_next_batch(self, batch_size):
        return self.get_next_batch_(batch_size)

class FiveK(Dataset):
    def __init__(self, path_img, loadSize):
        super().__init__()
        self.path_img = path_img
        self.data_img = sorted(os.listdir(path_img))
        self.loadSize = loadSize
        self.resize = torchvision.transforms.Resize((512, 512))
        self.is_test = ('test' in path_img) or ('val' in path_img)

    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, idx):
        name = str(os.path.join(self.path_img, self.data_img[idx]))
        img = read_tiff16(name)
        img = torch.tensor(img).permute(2,0,1)
        C, H, W = img.shape
        width = min(H, W)

        rnd_h = 0
        rnd_w = 0

        img = img[ :, rnd_h:rnd_h + width, rnd_w:rnd_w + width]
        return self.resize(img), name

class LoadImagesAndLabelsNormalizeHR(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        super(LoadImagesAndLabelsNormalizeHR, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:

            img, labels = self.load_mosaic(index)
            shapes = None

            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:

            img, (h0, w0), (h, w) = self.load_image(index)
            img_hr = img.copy()

            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img) / 255.

        img_hr = img_hr.transpose((2, 0, 1))[::-1]
        img_hr = np.ascontiguousarray(img_hr) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, torch.from_numpy(img_hr)

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes, img_hr = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)

class LoadImagesAndLabelsNormalizeReplay(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        super(LoadImagesAndLabelsNormalizeReplay, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                                 cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)

    def __getitem__(self, index):

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:

            img, labels = self.load_mosaic(index)
            shapes = None

            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:

            img, (h0, w0), (h, w) = self.load_image(index)

            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)

        return im_list, label_list, path_list, shapes_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret

def img2label_paths_rod(img_paths, img_dir_name="images"):

    sa, sb = f'{os.sep}{img_dir_name}{os.sep}', f'{os.sep}labels{os.sep}'
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

class LoadImagesAndLabelsRODReplay(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        img_dir_name = "npy"
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)

                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        img_dir_name = t[0].split("/")[1]
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]

                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n') from e

        if 0 < limit < len(self.im_files):
            self.im_files = self.im_files[:limit]
            LOGGER.warning(f"Select {limit} images as training data!")

        self.label_files = img2label_paths_rod(self.im_files, img_dir_name)

        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache['version'] == self.cache_version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False

        nf, nm, ne, nc, n = cache.pop('results')
        if exists and LOCAL_RANK in {-1, 0}:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training.'

        [cache.pop(k) for k in ('hash', 'version', 'msgs')]
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training.'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys())

        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]

        n = len(self.shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(int)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = range(n)

        include_class = []
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:
                self.labels[i][:, 0] = 0

        if self.rect:

            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)

    @staticmethod
    def verify_image_label(args):

        im_file, lb_file, prefix = args
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []
        try:

            im = np.load(im_file)
            shape = im.shape
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'

            if os.path.isfile(lb_file):
                nf = 1
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb):
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:
                        lb = lb[i]
                        if segments:
                            segments = [segments[x] for x in i]
                        msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
                else:
                    ne = 1
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1
                lb = np.zeros((0, 5), dtype=np.float32)

            return im_file, lb, shape, segments, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):

        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f'{prefix}Scanning {path.parent / path.stem}...'
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(self.verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs
        x['version'] = self.cache_version
        try:
            np.save(path, x)
            path.with_suffix('.cache.npy').rename(path)
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')
        return x

    def load_image(self, i):

        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:

            if fn.exists():
                im = np.load(fn)
                im = im / np.percentile(im, 99)
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1]
            else:

                im = np.load(f)
                im = im / np.percentile(im, 99)
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1]
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def __getitem__(self, index):

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:

            img, labels = self.load_mosaic(index)
            shapes = None

            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:

            img, (h0, w0), (h, w) = self.load_image(index)

            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:

            img, labels = self.albumentations(img, labels)
            nl = len(labels)

            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)

        return im_list, label_list, path_list, shapes_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret

class LoadImagesAndLabelsROD(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        img_dir_name = "npy"
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)

                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        img_dir_name = t[0].split("/")[1]
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]

                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n') from e

        if 0 < limit < len(self.im_files):
            self.im_files = self.im_files[:limit]
            LOGGER.warning(f"Select {limit} images as training data!")

        self.label_files = img2label_paths_rod(self.im_files, img_dir_name)

        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache['version'] == self.cache_version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False

        nf, nm, ne, nc, n = cache.pop('results')
        if exists and LOCAL_RANK in {-1, 0}:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training.'

        [cache.pop(k) for k in ('hash', 'version', 'msgs')]
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training.'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys())

        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]

        n = len(self.shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(int)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = range(n)

        include_class = []
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:
                self.labels[i][:, 0] = 0

        if self.rect:

            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)

    @staticmethod
    def verify_image_label(args):

        im_file, lb_file, prefix = args
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []
        try:

            im = np.load(im_file)
            shape = im.shape
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'

            if os.path.isfile(lb_file):
                nf = 1
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb):
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:
                        lb = lb[i]
                        if segments:
                            segments = [segments[x] for x in i]
                        msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
                else:
                    ne = 1
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1
                lb = np.zeros((0, 5), dtype=np.float32)

            return im_file, lb, shape, segments, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):

        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f'{prefix}Scanning {path.parent / path.stem}...'
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(self.verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs
        x['version'] = self.cache_version
        try:
            np.save(path, x)
            path.with_suffix('.cache.npy').rename(path)
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')
        return x

    def load_image(self, i):

        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:

            if fn.exists():
                im = np.load(fn).astype(np.float32)
                im = im / np.percentile(im, 99)
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1]
            else:

                im = np.load(f).astype(np.float32)
                im = im / np.percentile(im, 99)
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1]
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def __getitem__(self, index):

        hyp = self.hyp

        img, (h0, w0), (h, w) = self.load_image(index)

        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

def restore_image(image, ori_image):
    ih, iw, _ = image.shape
    if isinstance(ori_image, (tuple, list)):
        h, w, _ = ori_image
    else:
        h, w, _ = ori_image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    dst_img = image[dh:dh + nh, dw:dw + nw, ::]

    dst_img = cv2.resize(dst_img, (w, h))

    return dst_img

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def show(x, title="a", format="HWC", is_last=True):
        if format == 'CHW':
            x = np.transpose(x, (1, 2, 0))
        plt.figure()
        plt.cla()
        plt.title(title)
        plt.imshow(x)
        if is_last:
            plt.show()
    data_dict = {'path': 'COCO/coco2017',
     'train': 'COCO/coco2017/train2017.txt',
     'val': 'COCO/coco2017/val2017.txt',
     'test': 'COCO/coco2017/test-dev2017.txt',
     'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
               29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
               48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
               55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
               62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
               69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
               76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'},
     'download': "from utils.general import download, Path\n\n\n# Download labels\nsegments = False  # segment or box labels\ndir = Path(yaml['path'])  # dataset root dir\nurl = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'\nurls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels\ndownload(urls, dir=dir.parent)\n\n# Download data\nurls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images\n        'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images\n        'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)\ndownload(urls, dir=dir / 'images', threads=3)\n",
     'nc': 80}

    batch_size = 4
    imgsz = 512
    val_path = 'COCO/coco2017/val2017.txt'
    dataset = LoadImagesAndLabelsRAWReplay(
        val_path,
        imgsz,
        batch_size,
        augment=False,
        limit=1000
    )
    print(len(dataset))
    print(dataset.get_next_batch_(batch_size))
    exit()
