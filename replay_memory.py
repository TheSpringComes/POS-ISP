import random

import numpy as np
import torch

from dataset import LoadImagesAndLabelsNormalizeReplay, LoadImagesAndLabelsRAWReplaySeg
from util import Dict, STATE_STEP_DIM, STATE_STOPPED_DIM

def create_input_tensor(batch):
    im_list, label_list, path_list, shapes_list, states_list = batch
    for i, lb in enumerate(label_list):
        lb[:, 0] = i
    return (
        torch.from_numpy(np.stack(im_list, 0)),
        torch.from_numpy(np.concatenate(label_list, 0)),
        path_list,
        shapes_list,
        torch.from_numpy(np.stack(states_list, 0)),
    )

class ReplayMemorySeg:
    def __init__(self, cfg, load, path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0, rect=False, image_weights=False, prefix="", limit=-1, data_name="lis", add_noise=False, brightness_range=None, noise_level=None, use_linear=False, data_dict=None, yaml_path="", mode="train"):
        self.cfg = cfg
        if data_name == "lis":
            self.dataset = LoadImagesAndLabelsRAWReplaySeg(
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
                add_noise=add_noise,
                brightness_range=brightness_range,
                noise_level=noise_level,
                use_linear=use_linear,
                yaml_path=yaml_path,
                mode=mode,
            )
        else:
            raise ValueError("ReplayMemory input data_name error!")
        self.image_pool = []
        self.target_pool_size = cfg.replay_memory_size
        self.batch_size = batch_size
        if load:
            self.load()

    def load(self):
        self.fill_pool()

    def get_initial_states(self, batch_size):
        states = np.zeros(shape=(batch_size, self.cfg.num_state_dim), dtype=np.float32)
        for k in range(batch_size):
            for i in range(len(self.cfg.filters)):
                states[k, -(i + 1)] = 0
        return states

    def fill_pool(self):
        while len(self.image_pool) < self.target_pool_size:
            im_list, shapes_list, resized_shapes_list, masks_list, img_list, cls_list, bboxes_list, batch_idx_list = self.dataset.get_next_batch(self.batch_size)
            for i in range(len(im_list)):
                self.image_pool.append(
                    Dict(
                        im=im_list[i],
                        shape=shapes_list[i],
                        resized_shape=resized_shapes_list[i],
                        mask=masks_list[i],
                        img=img_list[i],
                        cls=cls_list[i],
                        bbox=bboxes_list[i],
                        batch_idx=batch_idx_list[i],
                        state=self.get_initial_states(1)[0],
                    )
                )
        self.image_pool = self.image_pool[: self.target_pool_size]

    def get_feed_dict_and_states(self, batch_size):
        im_list, shapes_list, resized_shapes_list, masks_list, img_list, cls_list, bboxes_list, batch_idx_list, states = self.get_next_fake_batch(batch_size)
        z = self.get_noise(batch_size)
        return {
            "im_file": im_list,
            "ori_shape": shapes_list,
            "resized_shape": resized_shapes_list,
            "masks": masks_list,
            "img": img_list,
            "cls": cls_list,
            "bboxes": bboxes_list,
            "batch_idx": batch_idx_list,
            "state": states,
            "z": z,
        }

    def get_noise(self, batch_size):
        if self.cfg.z_type == "normal":
            return np.random.normal(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        if self.cfg.z_type == "uniform":
            return np.random.uniform(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        raise AssertionError(f"Unknown noise type: {self.cfg.z_type}")

    @staticmethod
    def records_to_images_and_states(batch):
        return (
            [x["im"] for x in batch],
            [x["shape"] for x in batch],
            [x["resized_shape"] for x in batch],
            [x["mask"] for x in batch],
            [x["img"] for x in batch],
            [x["cls"] for x in batch],
            [x["bbox"] for x in batch],
            [x["batch_idx"] for x in batch],
            [x["state"] for x in batch],
        )

    def get_next_fake_batch(self, batch_size):
        random.shuffle(self.image_pool)
        assert batch_size <= len(self.image_pool)
        batch = []
        while len(batch) < batch_size:
            if len(self.image_pool) == 0:
                self.fill_pool()
            record = self.image_pool.pop(0)
            if record.state[STATE_STOPPED_DIM] != 1:
                batch.append(record)
        return self.records_to_images_and_states(batch)

    def debug(self):
        tot_trajectory = sum(r.state[STATE_STEP_DIM] for r in self.image_pool)
        average_trajectory = 1.0 * tot_trajectory / len(self.image_pool)
        print("# Replay memory: size %d, avg. traj. %.2f" % (len(self.image_pool), average_trajectory))
        print("#--------------------------------------------")

class ReplayMemory:
    def __init__(self, cfg, load, path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0, rect=False, image_weights=False, prefix="", limit=-1, data_name="lod", add_noise=False, brightness_range=None, noise_level=None, use_linear=False, dataset=None):
        self.cfg = cfg
        if data_name == "lod":
            self.dataset = LoadImagesAndLabelsNormalizeReplay(
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
        else:
            raise ValueError("ReplayMemory supports only lod.")
        self.image_pool = []
        self.target_pool_size = cfg.replay_memory_size
        self.batch_size = batch_size
        if load:
            self.load()

    def load(self):
        self.fill_pool()

    def get_initial_states(self, batch_size):
        states = np.zeros(shape=(batch_size, self.cfg.num_state_dim), dtype=np.float32)
        for k in range(batch_size):
            for i in range(len(self.cfg.filters)):
                states[k, -(i + 1)] = 0
        return states

    def fill_pool(self):
        while len(self.image_pool) < self.target_pool_size:
            im_list, label_list, path_list, shapes_list = self.dataset.get_next_batch(self.batch_size)
            for i in range(len(im_list)):
                self.image_pool.append(
                    Dict(
                        im=im_list[i],
                        label=label_list[i],
                        path=path_list[i],
                        shape=shapes_list[i],
                        state=self.get_initial_states(1)[0],
                    )
                )
        self.image_pool = self.image_pool[: self.target_pool_size]

    def get_feed_dict_and_states(self, batch_size):
        images, labels, paths, shapes, states = self.get_next_fake_batch(batch_size)
        z = self.get_noise(batch_size)
        return {"im": images, "label": labels, "path": paths, "shape": shapes, "state": states, "z": z}

    def get_noise(self, batch_size):
        if self.cfg.z_type == "normal":
            return np.random.normal(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        if self.cfg.z_type == "uniform":
            return np.random.uniform(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        raise AssertionError(f"Unknown noise type: {self.cfg.z_type}")

    @staticmethod
    def records_to_images_and_states(batch):
        return [x["im"] for x in batch], [x["label"] for x in batch], [x["path"] for x in batch], [x["shape"] for x in batch], [x["state"] for x in batch]

    def get_next_fake_batch(self, batch_size):
        random.shuffle(self.image_pool)
        assert batch_size <= len(self.image_pool)
        batch = []
        while len(batch) < batch_size:
            if len(self.image_pool) == 0:
                self.fill_pool()
            record = self.image_pool.pop(0)
            if record.state[STATE_STOPPED_DIM] != 1:
                batch.append(record)
        return self.records_to_images_and_states(batch)

    def debug(self):
        tot_trajectory = sum(r.state[STATE_STEP_DIM] for r in self.image_pool)
        average_trajectory = 1.0 * tot_trajectory / len(self.image_pool)
        print("# Replay memory: size %d, avg. traj. %.2f" % (len(self.image_pool), average_trajectory))
        print("#--------------------------------------------")
