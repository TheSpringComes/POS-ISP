import copy
import importlib
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics import YOLO

import sys
sys.path.append("yolov3")

from agent import Agent
from replay_memory import ReplayMemorySeg
from util import Tee
from yolov3.utils.callbacks import Callbacks
from yolov3.utils.downloads import attempt_download
from yolov3.utils.general import check_dataset, check_img_size, colorstr, labels_to_class_weights
from yolov3.utils.loss import ComputeLossV13Seg
from yolov3.utils.torch_utils import torch_distributed_zero_first

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PosISP:
    def __init__(self, args, task="train_val"):
        train = task in ("train", "train_val")
        val = task == "train_val"
        if train:
            self.base_dir = os.path.join(args.save_dir_name, args.save_path)
            self.log_dir = os.path.join(self.base_dir, "logs")
            self.ckpt_dir = os.path.join(self.base_dir, "ckpt")
            self.image_dir = os.path.join(self.base_dir, "images")
            self.seq_dir = os.path.join(self.base_dir, "seq")
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.seq_dir, exist_ok=True)
            self.tee = Tee(os.path.join(self.log_dir, "log.txt"))
            self.writer = SummaryWriter(self.log_dir)
            shutil.copy("config.py", os.path.join(self.base_dir, "config.py"))
        cfg = importlib.import_module(args.cfg).cfg
        self.device = torch.device("cuda")
        hyp = args.hyp
        if isinstance(hyp, str):
            with open(hyp, errors="ignore") as f:
                hyp = yaml.safe_load(f)
        args.hyp = hyp.copy()
        data_dict = check_dataset(args.data_cfg)
        nc = int(data_dict["nc"])
        with torch_distributed_zero_first(LOCAL_RANK):
            attempt_download(args.weights)
        yolo_model = YOLO("yolo11n-seg.pt").to(self.device)
        train_path, val_path = data_dict["train"], data_dict["val"]
        gs = max(int(yolo_model.stride.max()), 32)
        args.imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)
        self.train_loader = ReplayMemorySeg(cfg, train, train_path, args.imgsz, args.batch_size, gs, single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0, rect=False, image_weights=False, prefix=colorstr("train: "), limit=-1, data_dict=data_dict, yaml_path=args.data_cfg, mode="train")
        if val:
            self.val_loader = ReplayMemorySeg(cfg, val, val_path, args.imgsz, args.batch_size, gs, single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0, rect=False, image_weights=False, prefix=colorstr("val: "), limit=-1, data_dict=data_dict, yaml_path=args.data_cfg, mode="val")
            self.val_loader = self.val_loader.get_feed_dict_and_states(8)
        core = yolo_model.model
        if hasattr(core, "model") and hasattr(core.model, "__getitem__"):
            nl = core.model[-1].nl
        elif hasattr(core, "head") and hasattr(core.head, "nl"):
            nl = core.head.nl
        else:
            nl = next((m.nl for m in core.modules() if hasattr(m, "nl")), None)
        hyp["box"] *= 3 / nl
        hyp["cls"] *= nc / 80 * 3 / nl
        hyp["obj"] *= (args.imgsz / 640) ** 2 * 3 / nl
        hyp["dfl"] = hyp["box"] * 0.2
        hyp["label_smoothing"] = 0.0
        core.nc = nc
        core.hyp = hyp
        core.names = data_dict["names"]
        core.class_weights = labels_to_class_weights(self.train_loader.dataset.labels, nc).to(self.device) * nc
        self.yolo_model = core.to(self.device)
        self.yolo = yolo_model
        self.agent = Agent(self.log_dir, cfg.filters).to(self.device)
        self.args = args
        self.cfg = cfg

    @staticmethod
    def compute_loss_batch(func, preds, targets, device):
        batch = preds[0][0].shape[0]
        total_loss = torch.zeros((batch, 1), device=device)
        for b in range(batch):
            pred_one = []
            for i in range(len(preds)):
                if i == 0:
                    pred_one.append([preds[0][j][b].unsqueeze(0) for j in range(len(preds[0]))])
                else:
                    pred_one.append(preds[i][b].unsqueeze(0).to(device))
            target_one = {}
            for key in targets.keys():
                target_one[key] = [targets[key][b]]
            loss, losses = func(pred_one, target_one)
            total_loss[b] += loss
        return total_loss

    def train(self):
        start_iter = 0
        if self.args.resume is not None:
            ckpt = torch.load(self.args.resume)
            self.agent.load_state_dict(ckpt["agent_model"])
            start_iter = ckpt["iter"]
        p_optimizer = torch.optim.Adam(self.agent.param_net.parameters(), lr=1e-4, betas=(0.9, 0.99))
        a_optimizer = torch.optim.Adam(self.agent.action_agent.parameters(), lr=1e-5, betas=(0.9, 0.99))
        compute_loss = ComputeLossV13Seg(self.yolo_model)
        callbacks = Callbacks()
        callbacks.run("on_train_start")
        self.agent.param_net.train()
        self.agent.action_agent.train()
        self.yolo_model.train()
        for _, value in self.yolo_model.named_parameters():
            value.requires_grad = False
        for module in self.yolo_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for iter_idx in tqdm(range(start_iter, self.cfg.max_iter_step + 1), desc="Training", total=self.cfg.max_iter_step + 1):
            feed_dict = self.train_loader.get_feed_dict_and_states(self.args.batch_size)
            p_optimizer.zero_grad()
            a_optimizer.zero_grad()
            imgs = torch.from_numpy(np.stack(feed_dict["img"], 0)) / 255.0
            imgs = imgs.to(self.device, non_blocking=True).float()
            agent_out = self.agent(imgs, iter_idx, writer=self.writer, save_path=self.args.save_path, is_val=False)
            retouch = torch.clip(agent_out["output"], 0.0, 1.0)
            surrogate = agent_out["selected_prob"]
            penalty = agent_out["penalty"].unsqueeze(1)
            detect_input_loss = self.compute_loss_batch(compute_loss, self.yolo_model(imgs), copy.deepcopy(feed_dict), self.device)
            detect_input_loss = torch.clip(detect_input_loss * self.cfg.seg_loss_weight, 0, 1.0)
            detect_retouch_loss = self.compute_loss_batch(compute_loss, self.yolo_model(retouch), copy.deepcopy(feed_dict), self.device)
            detect_retouch_loss = torch.clip(detect_retouch_loss * self.cfg.seg_loss_weight, 0, 1.0)
            reward = detect_input_loss.detach() - detect_retouch_loss
            if self.cfg.use_penalty:
                reward -= penalty
            param_loss = -torch.mean(reward)
            if param_loss.requires_grad:
                param_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.param_net.parameters(), 1e-4)
                p_optimizer.step()
            action_loss = torch.mean(surrogate * -reward.detach())
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.action_agent.parameters(), 1e-5)
            a_optimizer.step()
            if iter_idx % self.cfg.summary_freq == 0:
                self.writer.add_scalar("action_loss", action_loss, global_step=iter_idx)
                self.writer.add_scalar("param_loss", param_loss, global_step=iter_idx)
                self.writer.add_scalar("detect_loss", detect_retouch_loss.mean(), global_step=iter_idx)
            self.train_loader.fill_pool()
            if iter_idx % self.cfg.save_model_freq == 0:
                torch.save({"iter": iter_idx, "agent_model": self.agent.state_dict()}, os.path.join(self.ckpt_dir, f"PosISP_iter_{iter_idx}.pth"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--weights", type=str, default="../../pretrained/yolov3.pt")
    parser.add_argument("--yolo_cfg", type=str, default="yolov3/models/yolov3.yaml")
    parser.add_argument("--hyp", type=str, default="yolov3/data/hyps/hyp.scratch-low.yaml")
    parser.add_argument("--save_dir_name", type=str, default="experiments")
    parser.add_argument("--save_path", type=str, default="posisp")
    parser.add_argument("--data_name", type=str, default="lis", choices=["lis"])
    parser.add_argument("--data_cfg", type=str, default="yolov3/data/lis_raw_all.yaml")
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--use_linear", action="store_true", default=False)
    parser.add_argument("--bri_range", type=float, default=None, nargs="*")
    parser.add_argument("--noise_level", type=float, default=None)
    parser.add_argument("--runtime_penalty", action="store_true", default=False)
    parser.add_argument("--runtime_penalty_lambda", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model_weights", type=str, default="experiments/")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfg", type=str, default="config")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task", type=str, default="train")
    args = parser.parse_args()
    set_seed(args.seed)
    args.save_path = args.data_name + "-" + args.save_path
    PosISP(args, args.task).train()
