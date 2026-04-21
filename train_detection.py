import importlib
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append("yolov3")

from agent import Agent
from replay_memory import ReplayMemory, create_input_tensor
from util import Tee
from yolov3.models.yolo import Model
from yolov3.utils.callbacks import Callbacks
from yolov3.utils.downloads import attempt_download
from yolov3.utils.general import check_dataset, check_img_size, colorstr, intersect_dicts, labels_to_class_weights
from yolov3.utils.loss import ComputeLoss, ComputeLossBatch
from yolov3.utils.torch_utils import torch_distributed_zero_first

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
_DEBUG_LOG_PATH = "/home/jing/projects/SDI/POS-ISP/.cursor/debug-1d6553.log"
_DEBUG_SESSION_ID = "1d6553"


def _agent_debug_log(hypothesis_id, location, message, data=None, run_id="pre-fix"):
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data if data is not None else {},
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        try:
            print(
                f"[agent_debug_log_write_failed] path={_DEBUG_LOG_PATH} error={type(e).__name__}: {e}",
                file=sys.stderr,
            )
        except Exception:
            pass

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
            os.makedirs(self.base_dir, exist_ok=True)
            self.log_dir = os.path.join(self.base_dir, "logs")
            self.ckpt_dir = os.path.join(self.base_dir, "ckpt")
            self.image_dir = os.path.join(self.base_dir, "seq")
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)
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
            weights = attempt_download(args.weights)
        ckpt = torch.load(weights, map_location="cpu")
        yolo_model = Model(args.yolo_cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(self.device)
        exclude = ["anchor"]
        csd = intersect_dicts(ckpt["model"].float().state_dict(), yolo_model.state_dict(), exclude=exclude)
        yolo_model.load_state_dict(csd, strict=False)
        train_path, val_path = data_dict["train"], data_dict["val"]
        if task == "test":
            val_path = data_dict["test"]
        gs = max(int(yolo_model.stride.max()), 32)
        args.imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)
        self.train_loader = ReplayMemory(cfg, train, train_path, args.imgsz, args.batch_size, gs, single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0, rect=False, image_weights=False, prefix=colorstr("train: "), limit=-1, add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range, noise_level=args.noise_level, use_linear=args.use_linear)
        if val:
            self.val_loader = ReplayMemory(cfg, val, val_path, args.imgsz, args.batch_size, gs, single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0, rect=False, image_weights=False, prefix=colorstr("val: "), limit=-1, add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range, noise_level=args.noise_level, use_linear=args.use_linear)
            self.val_loader = self.val_loader.get_feed_dict_and_states(8)
        nl = yolo_model.model[-1].nl
        hyp["box"] *= 3 / nl
        hyp["cls"] *= nc / 80 * 3 / nl
        hyp["obj"] *= (args.imgsz / 640) ** 2 * 3 / nl
        hyp["label_smoothing"] = 0.0
        yolo_model.nc = nc
        yolo_model.hyp = hyp
        yolo_model.class_weights = labels_to_class_weights(self.train_loader.dataset.labels, nc).to(self.device) * nc
        yolo_model.names = data_dict["names"]
        self.yolo_model = yolo_model.to(self.device)
        self.agent = Agent(self.log_dir, cfg.filters).to(self.device)
        self.args = args
        self.cfg = cfg

        # print(self.yolo_model)
        n_params = sum(p.numel() for p in self.yolo_model.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")
        # print(self.agent)
        n_params = sum(p.numel() for p in self.agent.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")

        # region agent log
        _agent_debug_log(
            "H7",
            "train_detection.py:PosISP.__init__",
            "posisp initialization complete",
            data={
                "task": str(task),
                "train_enabled": bool(train),
                "val_enabled": bool(val),
                "device": str(self.device),
                "batch_size": int(args.batch_size),
                "imgsz": int(args.imgsz),
                "save_path": str(args.save_path),
                "cwd": os.getcwd(),
                "file": __file__,
            },
        )
        # endregion

    @staticmethod
    def compute_loss_batch(func, preds, targets, device):
        batch = preds[0].shape[0]
        lclss = torch.zeros((batch, 1), device=device)
        lboxs = torch.zeros((batch, 1), device=device)
        lobjs = torch.zeros((batch, 1), device=device)
        for b in range(batch):
            pred_one = [preds[i][b].unsqueeze(0).to(device) for i in range(len(preds))]
            target_one = targets[b]
            target_one[:, 0] = 0
            lbox, lobj, lcls = func(pred_one, target_one.to(device))
            lboxs[b], lobjs[b], lclss[b] = lbox, lobj, lcls
        return lboxs + lobjs + lclss, torch.cat((lboxs, lobjs, lclss)).detach()

    @staticmethod
    def _tensor_grad_meta(x):
        if isinstance(x, torch.Tensor):
            return {
                "requires_grad": bool(x.requires_grad),
                "shape": list(x.shape),
                "grad_fn": type(x.grad_fn).__name__ if x.grad_fn is not None else None,
            }
        if isinstance(x, (list, tuple)):
            return [PosISP._tensor_grad_meta(i) for i in list(x)[:5]]
        return {"type": str(type(x))}

    def train(self):
        start_iter = 0
        if self.args.resume is not None:
            ckpt = torch.load(self.args.resume)
            self.agent.load_state_dict(ckpt["agent_model"])
            start_iter = ckpt["iter"]
        p_optimizer = torch.optim.Adam(self.agent.param_net.parameters(), lr=1e-4, betas=(0.9, 0.99))
        a_optimizer = torch.optim.Adam(self.agent.action_agent.parameters(), lr=1e-5, betas=(0.9, 0.99))
        compute_loss = ComputeLoss(self.yolo_model)
        compute_loss_batch = ComputeLossBatch(self.yolo_model, reduction="mean")
        callbacks = Callbacks()
        callbacks.run("on_train_start")
        # region agent log
        _agent_debug_log(
            "H8",
            "train_detection.py:PosISP.train",
            "train loop setup complete",
            data={
                "start_iter": int(start_iter),
                "max_iter_step": int(self.cfg.max_iter_step),
                "summary_freq": int(self.cfg.summary_freq),
                "save_model_freq": int(self.cfg.save_model_freq),
                "batch_size": int(self.args.batch_size),
            },
        )
        # endregion
        mloss_param, mloss_action, mloss_detect = 0.0, 0.0, 0.0
        for iter_idx in tqdm(range(start_iter, self.cfg.max_iter_step + 1), desc="Training", total=self.cfg.max_iter_step + 1):
            self.agent.param_net.train()
            self.agent.action_agent.train()
            self.yolo_model.train()
            for _, value in self.yolo_model.named_parameters():
                value.requires_grad = False
            for module in self.yolo_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            feed_dict = self.train_loader.get_feed_dict_and_states(self.args.batch_size)
            imgs, targets, paths, shapes, states = create_input_tensor((feed_dict["im"], feed_dict["label"], feed_dict["path"], feed_dict["shape"], feed_dict["state"]))
            p_optimizer.zero_grad()
            a_optimizer.zero_grad()
            imgs = imgs.to(self.device, non_blocking=True).float()
            agent_out = self.agent(imgs, iter_idx, writer=self.writer, save_path=self.base_dir)
            retouch = agent_out["output"]
            surrogate = agent_out["selected_prob"]
            penalty = agent_out["penalty"].unsqueeze(1)
            selected_orders = agent_out.get("selected_orders", [])
            effective_actions = []
            for order in selected_orders:
                active_count = 0
                for action in order:
                    if action == self.agent.eos_token:
                        break
                    active_count += 1
                effective_actions.append(active_count)
            all_eos = bool(effective_actions) and max(effective_actions) == 0
            if all_eos:
                # region agent log
                _agent_debug_log(
                    "H12",
                    "train_detection.py:PosISP.train:all_eos_batch",
                    "batch has zero non-eos actions before detection loss",
                    data={
                        "iter_idx": int(iter_idx),
                        "effective_actions": [int(x) for x in effective_actions],
                        "retouch": self._tensor_grad_meta(retouch),
                        "surrogate": self._tensor_grad_meta(surrogate),
                    },
                )
                # endregion
            if iter_idx < 5:
                # region agent log
                _agent_debug_log(
                    "H1",
                    "train_detection.py:PosISP.train:after_agent_forward",
                    "agent output grad and action trace",
                    data={
                        "iter_idx": int(iter_idx),
                        "grad_enabled": bool(torch.is_grad_enabled()),
                        "imgs": self._tensor_grad_meta(imgs),
                        "retouch": self._tensor_grad_meta(retouch),
                        "surrogate": self._tensor_grad_meta(surrogate),
                        "penalty": self._tensor_grad_meta(penalty),
                        "effective_actions": [int(x) for x in effective_actions],
                        "first_selected_orders": [list(map(int, o[:8])) for o in selected_orders[:3]],
                        "all_eos": bool(all_eos),
                    },
                )
                # endregion
            if effective_actions and max(effective_actions) > 0 and (not retouch.requires_grad):
                # region agent log
                _agent_debug_log(
                    "H2",
                    "train_detection.py:PosISP.train:retouch_grad_break",
                    "retouch has no grad despite non-eos action",
                    data={
                        "iter_idx": int(iter_idx),
                        "effective_actions": [int(x) for x in effective_actions],
                        "retouch": self._tensor_grad_meta(retouch),
                    },
                )
                # endregion
            pred_input = self.yolo_model(imgs)
            detect_input_loss, _ = self.compute_loss_batch(compute_loss_batch, pred_input, feed_dict["label"], self.device)
            detect_input_loss = torch.clip(detect_input_loss, 0, 1.0)
            pred_retouch = self.yolo_model(retouch)
            _, detect_retouch_loss_items = compute_loss(pred_retouch, targets.to(self.device))
            detect_retouch_loss, _ = self.compute_loss_batch(compute_loss_batch, pred_retouch, feed_dict["label"], self.device)
            detect_retouch_loss = torch.clip(detect_retouch_loss, 0, 1.0)
            if iter_idx < 5 or (not detect_retouch_loss.requires_grad):
                # region agent log
                _agent_debug_log(
                    "H3",
                    "train_detection.py:PosISP.train:after_detection_loss",
                    "detection loss gradient connectivity",
                    data={
                        "iter_idx": int(iter_idx),
                        "pred_input": self._tensor_grad_meta(pred_input),
                        "pred_retouch": self._tensor_grad_meta(pred_retouch),
                        "detect_input_loss": self._tensor_grad_meta(detect_input_loss),
                        "detect_retouch_loss": self._tensor_grad_meta(detect_retouch_loss),
                        "detect_retouch_loss_items_shape": list(detect_retouch_loss_items.shape),
                    },
                )
                # endregion
            reward = detect_input_loss.detach() - detect_retouch_loss
            if self.cfg.use_penalty:
                reward -= penalty
            param_loss = -torch.mean(reward)
            if iter_idx < 5 or (not param_loss.requires_grad):
                # region agent log
                _agent_debug_log(
                    "H4",
                    "train_detection.py:PosISP.train:before_param_backward",
                    "reward and param loss grad status",
                    data={
                        "iter_idx": int(iter_idx),
                        "reward": self._tensor_grad_meta(reward),
                        "param_loss": self._tensor_grad_meta(param_loss),
                        "detect_input_loss": self._tensor_grad_meta(detect_input_loss),
                        "detect_retouch_loss": self._tensor_grad_meta(detect_retouch_loss),
                        "penalty": self._tensor_grad_meta(penalty),
                        "all_eos": bool(all_eos),
                    },
                )
                # endregion
            if param_loss.requires_grad:
                try:
                    param_loss.backward()
                except Exception as e:
                    # region agent log
                    _agent_debug_log(
                        "H5",
                        "train_detection.py:PosISP.train:param_loss_backward_exception",
                        "param_loss backward raised exception",
                        data={
                            "iter_idx": int(iter_idx),
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "param_loss": self._tensor_grad_meta(param_loss),
                            "reward": self._tensor_grad_meta(reward),
                        },
                    )
                    # endregion
                    raise
                torch.nn.utils.clip_grad_norm_(self.agent.param_net.parameters(), 1e-4)
                p_optimizer.step()
            else:
                # region agent log
                _agent_debug_log(
                    "H6",
                    "train_detection.py:PosISP.train:skip_param_backward",
                    "skip param optimizer step because param_loss is detached",
                    data={
                        "iter_idx": int(iter_idx),
                        "param_loss": self._tensor_grad_meta(param_loss),
                        "reward": self._tensor_grad_meta(reward),
                        "all_eos": bool(all_eos),
                    },
                    run_id="post-fix",
                )
                # endregion
            mloss_param = (mloss_param * iter_idx + param_loss.item()) / (iter_idx + 1)
            action_loss = torch.mean(surrogate * -reward.detach())
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.action_agent.parameters(), 1e-4)
            a_optimizer.step()
            mloss_action = (mloss_action * iter_idx + action_loss.item()) / (iter_idx + 1)
            mloss_detect = (mloss_detect * iter_idx + detect_retouch_loss_items.cpu().numpy()) / (iter_idx + 1)
            if iter_idx % self.cfg.summary_freq == 0:
                self.writer.add_scalar("action_loss", action_loss, global_step=iter_idx)
                self.writer.add_scalar("param_loss", param_loss, global_step=iter_idx)
                self.writer.add_scalar("detect_loss", detect_retouch_loss.mean(), global_step=iter_idx)
            self.train_loader.fill_pool()
            if iter_idx % self.cfg.save_model_freq == 0:
                torch.save({"iter": iter_idx, "agent_model": self.agent.state_dict()}, os.path.join(self.ckpt_dir, f"PosISP_iter_{iter_idx}.pth"))

if __name__ == "__main__":
    import argparse

    # region agent log
    print(
        f"[agent-debug-session={_DEBUG_SESSION_ID}] file={__file__} cwd={os.getcwd()} log={_DEBUG_LOG_PATH}",
        file=sys.stderr,
    )
    # endregion
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--weights", type=str, default="../../pretrained/yolov3.pt")
    parser.add_argument("--yolo_cfg", type=str, default="yolov3/models/yolov3.yaml")
    parser.add_argument("--hyp", type=str, default="yolov3/data/hyps/hyp.scratch-low.yaml")
    parser.add_argument("--save_dir_name", type=str, default="experiments")
    parser.add_argument("--save_path", type=str, default="posisp")
    parser.add_argument("--data_name", type=str, default="lod", choices=["lod", "coco"])
    parser.add_argument("--data_cfg", type=str, default="yolov3/data/lod.yaml")
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
    # region agent log
    _agent_debug_log(
        "H9",
        "train_detection.py:__main__",
        "train_detection entrypoint invoked",
        data={
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "pid": int(os.getpid()),
            "file": __file__,
        },
    )
    # endregion
    set_seed(args.seed)
    args.save_path = args.data_name + "-" + args.save_path
    args.add_noise = False
    args.bri_range = None
    args.use_linear = False
    PosISP(args, args.task).train()
