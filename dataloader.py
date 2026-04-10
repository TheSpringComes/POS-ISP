import os

import numpy as np
import torch
from torch.utils.data import DataLoader, distributed

import sys
sys.path.append("yolov3")

from yolov3.utils.dataloaders import InfiniteDataLoader, seed_worker
from yolov3.utils.general import LOGGER
from yolov3.utils.torch_utils import torch_distributed_zero_first

from dataset import KITTI, LoadImagesAndLabelsNormalize

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"

def create_dataloader_kitti_pair(batch_size, is_train=True, workers=8):
    dataset = KITTI(is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=is_train, num_workers=0)

def _build_loader(dataset, batch_size, rank, workers, shuffle, image_weights, collate_fn, seed):
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset

def create_dataloader_real(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0, rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix="", shuffle=False, seed=0, limit=-1, **kwargs):
    if rect and shuffle:
        LOGGER.warning("WARNING --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabelsNormalize(
            path,
            imgsz,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            limit=limit,
        )
    collate_fn = LoadImagesAndLabelsNormalize.collate_fn4 if quad else LoadImagesAndLabelsNormalize.collate_fn
    return _build_loader(dataset, batch_size, rank, workers, shuffle, image_weights, collate_fn, seed)

def get_noise(batch_size, z_type="uniform", z_dim=27):
    if z_type == "normal":
        return np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
    if z_type == "uniform":
        return np.random.uniform(0, 1, [batch_size, z_dim]).astype(np.float32)
    raise AssertionError(f"Unknown noise type: {z_type}")

def get_initial_states(batch_size, num_state_dim, filters_number):
    states = np.zeros(shape=(batch_size, num_state_dim), dtype=np.float32)
    for k in range(batch_size):
        for i in range(filters_number):
            states[k, -(i + 1)] = 0
    return states
