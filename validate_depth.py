import argparse
import importlib
import os
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent
YOLOV3_ROOT = PROJECT_ROOT / 'yolov3'
for path in (PROJECT_ROOT, YOLOV3_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dataloader import create_dataloader_kitti_pair
from depth import models
from agent import Agent
from util import Tee, save_img
from utils.general import TQDM_BAR_FORMAT, check_yaml, increment_path, print_args
from utils.torch_utils import select_device

disp_net = None

def load_disp_net(device, weights_path):
    global disp_net
    if disp_net is not None:
        return disp_net
    disp_net = models.DispResNet(18, False).to(device)
    weights = torch.load(weights_path, map_location=device)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    for param in disp_net.parameters():
        param.requires_grad = False
    return disp_net

def colorize_depth(depth, vmin=None, vmax=None, cmap='turbo', invalid_val=0.0):
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().float().numpy()
    depth = depth.copy()
    valid = depth > invalid_val
    if valid.sum() == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if vmin is None:
        vmin = np.percentile(depth[valid], 2)
    if vmax is None:
        vmax = np.percentile(depth[valid], 98)
    depth = np.clip(depth, vmin, vmax)
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    colored = cm.get_cmap(cmap)(depth_norm)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored

def densify_depth_for_vis(depth):
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().float().numpy()
    valid_mask = (depth > 0).astype(np.uint8)
    if valid_mask.sum() == 0:
        return depth
    depth_filled = depth.copy().astype(np.float32)
    invalid_mask = (valid_mask == 0).astype(np.uint8)
    depth_norm = depth_filled.copy()
    maxv = max(depth_norm.max(), 1e-6)
    depth_norm = (depth_norm / maxv * 255).astype(np.uint8)
    inpainted = cv2.inpaint(depth_norm, invalid_mask, 3, cv2.INPAINT_NS)
    return inpainted.astype(np.float32) / 255.0 * maxv

def normalize_image(img):
    return (img - 0.45) / 0.225

@torch.no_grad()
def compute_errors(gt, pred):
    _, height, width = gt.size()
    crop_mask = gt[0] != gt[0]
    y1, y2 = int(0.40810811 * height), int(0.99189189 * height)
    x1, x2 = int(0.03594771 * width), int(0.96405229 * width)
    crop_mask[y1:y2, x1:x2] = True
    max_depth = 80.0
    metrics_per_img = []
    pred_img = None

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth) & crop_mask
        if valid.sum() == 0:
            metrics_per_img.append(torch.full((8,), float("nan"), device=current_gt.device))
            pred_img = torch.zeros_like(current_pred)
            continue

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)
        factor = torch.median(valid_gt) / torch.median(valid_pred)
        valid_pred_scaled = valid_pred * factor
        thresh = torch.max(valid_gt / valid_pred_scaled, valid_pred_scaled / valid_gt)
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()
        abs_diff = torch.mean(torch.abs(valid_gt - valid_pred_scaled))
        abs_rel = torch.mean(torch.abs(valid_gt - valid_pred_scaled) / valid_gt)
        sq_rel = torch.mean(((valid_gt - valid_pred_scaled) ** 2) / valid_gt)
        rmse = torch.sqrt(torch.mean((valid_gt - valid_pred_scaled) ** 2))
        rmslog = torch.sqrt(torch.mean((torch.log(valid_gt) - torch.log(valid_pred_scaled)) ** 2))
        metrics_per_img.append(torch.stack([abs_diff, abs_rel, sq_rel, a1, a2, a3, rmse, rmslog]))
        pred_img = current_pred * factor

    return torch.stack(metrics_per_img, dim=0), pred_img

@torch.no_grad()
def sup_error_cal(data_in, data_gt):
    output_disp = disp_net(normalize_image(data_in))
    output_depth = 1 / output_disp.squeeze(1)
    if data_gt.nelement() != output_depth.nelement():
        _, height, width = data_gt.size()
        output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [height, width]).squeeze(1)
    metrics, pred_depth = compute_errors(data_gt, output_depth)
    pred_depth = torch.clamp(pred_depth, 0.0, 80.0)
    return metrics, pred_depth.unsqueeze(0)

def build_isp_model(device, cfg_file, isp_weights):
    cfg = importlib.import_module(cfg_file).cfg
    isp_model = Agent(str(PROJECT_ROOT), cfg.filters)
    if 'xxx' not in isp_weights:
        isp_model.load_state_dict(torch.load(isp_weights, map_location=device)['agent_model'])
    isp_model.to(device)
    isp_model.eval()
    return isp_model, [flt.name for flt in isp_model.filter_arr]

def run(
    data,
    batch_size=1,
    device='',
    save_txt=False,
    project=PROJECT_ROOT / 'runs',
    name='validate_depth',
    exist_ok=False,
    half=False,
    isp_model='Agent',
    isp_weights='experiments/xxx.pth',
    save_image=False,
    save_param=False,
    cfg_file='config',
    depth_weights=None,
    **_,
):
    if isp_model != 'Agent':
        raise ValueError(f"not support {isp_model}")

    device = select_device(device, batch_size=batch_size)
    load_disp_net(device, depth_weights)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    Tee(os.path.join(save_dir, 'val_log.txt'))

    isp_model, filter_names = build_isp_model(device, cfg_file, isp_weights)
    dataloader = create_dataloader_kitti_pair(batch_size=1, is_train=False)
    cuda = device.type != 'cpu'

    os.makedirs(os.path.join(save_dir, "img_results"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img_input"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img_depth_pred"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img_depth_gt"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img_depth_gt_dense_vis"), exist_ok=True)
    if save_param:
        os.makedirs(os.path.join(save_dir, "param_results"), exist_ok=True)

    metric_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3', 'rmse', 'rmslog']
    metric_sum = {name: 0.0 for name in metric_names}
    count = 0

    with open(os.path.join(save_dir, "records.txt"), "w+", encoding="utf-8") as handle:
        handle.write(",".join(filter_names) + "\n")
        pbar = tqdm(dataloader, bar_format=TQDM_BAR_FORMAT)

        for batch_i, (im, gt, paths) in enumerate(pbar):
            if cuda:
                im = im.to(device, non_blocking=True)
                gt = gt.to(device)
            im = im.half() if half else im.float()

            retouch = im
            if 'xxx' not in isp_weights:
                agent_out = isp_model.inference(im)
                retouch = agent_out['output']

            metrics, output_depth = sup_error_cal(retouch, gt)
            path = paths[0]

            if batch_i < 1000:
                save_img(im, path, os.path.join(save_dir, "img_input"), None, "CHW", False)
                save_img(retouch, path, os.path.join(save_dir, "img_results"), None, "CHW", False)

                stem = Path(path).stem
                pred_depth_2d = output_depth[0]
                pred_vis = colorize_depth(pred_depth_2d, vmin=0.1, vmax=80.0, cmap='turbo')
                cv2.imwrite(
                    os.path.join(save_dir, "img_depth_pred", f"{stem}.png"),
                    cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR),
                )

                gt_depth_2d = gt[0]
                gt_vis = colorize_depth(gt_depth_2d, vmin=0.1, vmax=80.0, cmap='turbo')
                cv2.imwrite(
                    os.path.join(save_dir, "img_depth_gt", f"{stem}.png"),
                    cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR),
                )

                gt_dense_vis = densify_depth_for_vis(gt_depth_2d)
                gt_dense_vis = colorize_depth(gt_dense_vis, vmin=0.1, vmax=80.0, cmap='turbo')
                cv2.imwrite(
                    os.path.join(save_dir, "img_depth_gt_dense_vis", f"{stem}.png"),
                    cv2.cvtColor(gt_dense_vis, cv2.COLOR_RGB2BGR),
                )

            for metric_name, score in zip(metric_names, metrics[0]):
                metric_sum[metric_name] += score.item()
            count += 1

    print("\n===== Final Evaluation Metrics =====")
    for metric_name in metric_names:
        avg_score = metric_sum[metric_name] / count
        print(f"{metric_name:8s}: {avg_score:.6f}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default=PROJECT_ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='validate_depth', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--isp_model', default='Agent', help='isp model')
    parser.add_argument('--isp_weights', default='experiments/xxx.pth', help='isp weights')
    parser.add_argument('--data', type=str, default=YOLOV3_ROOT / 'data' / 'lod.yaml', help='dataset.yaml path')
    parser.add_argument('--save_image', action='store_true', help='save image results')
    parser.add_argument('--save_param', action='store_true', help='save parameter results')
    parser.add_argument('--cfg_file', type=str, default='config', help='config file')
    parser.add_argument('--depth_weights', type=str, default=str(PROJECT_ROOT / 'depth' / 'weight' / 'dispnet_model_best.pth.tar'), help='depth backbone weights')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt

def main(opt):
    if opt.task not in ('train', 'val', 'test'):
        raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test")')
    run(**vars(opt))

if __name__ == '__main__':
    main(parse_opt())
