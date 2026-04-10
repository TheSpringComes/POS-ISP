import importlib
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import create_dataloader_kitti_pair
from depth.models import DispResNet
from agent import Agent
from util import Tee

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_image(img):
    return (img - 0.45) / 0.225

class PosISP:
    def __init__(self, args, task="train_val"):
        train = task in ("train", "train_val")
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
        self.device = torch.device("cuda")
        self.train_loader = create_dataloader_kitti_pair(batch_size=args.batch_size)
        self.args = args
        self.cfg = importlib.import_module(args.cfg).cfg
        self.agent = Agent(self.log_dir, self.cfg.filters).to(self.device)
        self.disp_net = DispResNet(18, False).to(self.device)
        weights = torch.load(args.depth_weights, map_location=self.device)
        self.disp_net.load_state_dict(weights["state_dict"])
        self.disp_net.eval()
        for param in self.disp_net.parameters():
            param.requires_grad = False

    def _sup_error_cal(self, data_in, data_gt):
        output_disp = self.disp_net(normalize_image(data_in))
        output_depth = 1 / output_disp.squeeze(1)
        if data_gt.nelement() != output_depth.nelement():
            b, h, w = data_gt.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)
        return torch.mean(torch.abs(output_depth - data_gt), dim=(1, 2))

    def train(self):
        start_iter = 0
        if self.args.resume is not None:
            ckpt = torch.load(self.args.resume)
            self.agent.load_state_dict(ckpt["agent_model"])
            start_iter = ckpt["iter"]
        p_optimizer = torch.optim.Adam(self.agent.param_net.parameters(), lr=1e-4, betas=(0.9, 0.99))
        a_optimizer = torch.optim.Adam(self.agent.action_agent.parameters(), lr=1e-5, betas=(0.9, 0.99))
        iter_idx = start_iter
        pbar = tqdm(total=self.cfg.max_iter_step + 1, desc="Training")
        while iter_idx < self.cfg.max_iter_step + 1:
            for data in self.train_loader:
                self.agent.param_net.train()
                self.agent.action_agent.train()
                imgs, gts, paths = data
                imgs = imgs.to(self.device, non_blocking=True).float()
                gts = gts.to(self.device, non_blocking=True).float()
                p_optimizer.zero_grad()
                a_optimizer.zero_grad()
                agent_out = self.agent(imgs, iter_idx, writer=self.writer, save_path=self.base_dir)
                retouch = torch.clamp(agent_out["output"], 0.0, 1.0)
                surrogate = agent_out["selected_prob"]
                input_loss_items = self._sup_error_cal(imgs, gts).unsqueeze(1)
                retouch_loss_items = self._sup_error_cal(retouch, gts).unsqueeze(1)
                reward = input_loss_items.detach() - retouch_loss_items
                param_loss = torch.mean(retouch_loss_items)
                param_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.param_net.parameters(), 1e-4)
                p_optimizer.step()
                action_loss = torch.mean(surrogate * -reward.detach())
                action_loss.backward()
                a_optimizer.step()
                if iter_idx % self.cfg.summary_freq == 0:
                    self.writer.add_scalar("action_loss", action_loss, global_step=iter_idx)
                    self.writer.add_scalar("param_loss", param_loss, global_step=iter_idx)
                    self.writer.add_scalar("detect_loss", retouch_loss_items.mean(), global_step=iter_idx)
                if iter_idx % self.cfg.save_model_freq == 0:
                    torch.save({"iter": iter_idx, "agent_model": self.agent.state_dict()}, os.path.join(self.ckpt_dir, f"PosISP_iter_{iter_idx}.pth"))
                iter_idx += 1
                pbar.update(1)
                if iter_idx > self.cfg.max_iter_step:
                    break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--weights", type=str, default="depth_weights.pth.tar")
    parser.add_argument("--depth_weights", type=str, default="depth/weight/dispnet_model_best.pth.tar")
    parser.add_argument("--save_dir_name", type=str, default="experiments")
    parser.add_argument("--save_path", type=str, default="posisp")
    parser.add_argument("--data_name", type=str, default="kitti", choices=["kitti"])
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
