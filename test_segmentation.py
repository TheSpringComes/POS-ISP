import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import importlib

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent
YOLOV13_ROOT = PROJECT_ROOT / 'yolov13'
for path in (PROJECT_ROOT, YOLOV13_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
ROOT = YOLOV13_ROOT

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.ops import (LOGGER, TQDM_BAR_FORMAT, Profile, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, scale_boxes, xywh2xyxy, xyxy2xywh, check_dataset)
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from util import Tee
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import callbacks
from agent import Agent
from util import Tee, save_img

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})

def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

import math

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    if isinstance(imgsz, int):
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:
        imgsz = list(imgsz)
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

@smart_inference_mode()
def run(
        data,
        weights=None,
        batch_size=32,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=300,
        task='val',
        device='',
        workers=8,
        single_cls=False,
        augment=False,
        verbose=False,
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        save_json=False,
        project=ROOT / 'runs/val',
        name='exp',
        exist_ok=False,
        half=False,
        dnn=False,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=callbacks.get_default_callbacks(),
        compute_loss=None,
        isp_model=None,
        isp_weights=None,
        z_type="uniform",
        z_dim=16+3,
        steps=5,
        num_state_dim=3,
        data_name="lis",
        add_noise=False,
        bri_range=None,
        noise_level=None,
        save_image=False,
        use_linear=False,
        pipeline=None,
        save_param=False,
        cfg_file=None,
):
    training = model is not None and isp_model is not None
    if training:
        device, pt, jit, engine = next(model.parameters()).device, True, False, False
        half &= device.type != 'cpu'
        model.half() if half else model.float()
    else:
        device = select_device(device, batch=batch_size)

        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        Tee(os.path.join(save_dir, 'val_log.txt'))
        for k, v in locals().items():
            print(k, ":", v)

        model_weights = weights[0] if isinstance(weights, (list, tuple)) else weights
        model = YOLO(model_weights).to(device)
        stride = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=stride)
        batch_size = 1

        data = check_dataset(data)
        if isp_model == "Agent":
            try:
                cfg = importlib.import_module(cfg_file).cfg
            except Exception as e:
                print(e)
                print(f"don't support {cfg_file}!")
            isp_model = Agent(str(PROJECT_ROOT), cfg.filters)
            if 'xxx' not in isp_weights:
                isp_model.load_state_dict(torch.load(isp_weights, map_location=device)['agent_model'])
                print('loaded!!!!!!!!!!!!!!')
            isp_model.to(device)
            filter_name = [x.name for x in isp_model.filter_arr]
        else:
            raise ValueError(f"not support {isp_model}")

    model.eval()
    isp_model.eval()
    cuda = device.type != 'cpu'
    is_coco = False
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    if not training:
        pad, rect = (0.0, False)
        pad, rect = (0.0, False)
        task = 'val'
        data[task] = data[task].replace('/yolov13/', '/')
        data_dict = check_det_dataset(opt.data)
        img_path = data_dict[task]
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
        cfg.imgsz = 512
        cfg.task = "segment"
        dataset = build_yolo_dataset(cfg, img_path=img_path, batch=8, rect=False, data=data_dict, mode=task)
        dataloader = build_dataloader(dataset, batch=1, workers=0, shuffle=False)
        print("padding: ", pad, "tesing with rectangular:", rect)
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names

    os.makedirs(os.path.join(save_dir, "img_results"))

    if save_param:
        param_save_dir = os.path.join(save_dir, "param_results")
        os.makedirs(param_save_dir)

    dataset = build_yolo_dataset(cfg, img_path=img_path, batch=1, rect=False, data=data_dict, mode=task)
    dataloader = build_dataloader(dataset, batch=1, workers=0,shuffle=False)

    from types import SimpleNamespace
    from ultralytics.models.yolo.segment.val import SegmentationValidator

    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    from ultralytics.models.yolo.segment import SegmentationValidator
    args = dict(model="yolov11-seg.pt", data=opt.data)
    args["plots"] = True
    args["project"] = str(save_dir.parent)
    args["name"] = str(save_dir.name)
    args["exist_ok"] = True

    validator = SegmentationValidator(args=args)
    dt = Profile(), Profile(), Profile()
    pbar = tqdm(dataloader, bar_format=TQDM_BAR_FORMAT)
    model = model

    validator.run_callbacks("on_val_start")
    validator.device = model.device
    validator.args.batch = 1
    validator.data = check_det_dataset(validator.args.data)
    validator.stride = model.stride
    model.eval()

    validator.init_metrics(de_parallel(model))
    validator.save_dir = save_dir

    for batch_i, (batch) in enumerate(pbar):
        validator.run_callbacks('on_val_batch_start')
        im = batch['img']
        for k in ["batch_idx", "cls", "bboxes", "masks"]:
            batch[k] = batch[k].to('cuda')
        im = im.to(device, non_blocking=True) / 255.
        im = im.float()
        nb, _, height, width = im.shape
        retouch = im
        agent_out = isp_model.inference(im)
        retouch = agent_out['output']
        retouch = torch.clip(retouch, 0.0, 1.0)
        validator.batch_i = batch_i
        with dt[0]:
            batch = validator.preprocess(batch)
        with dt[1]:
            preds = model.model((retouch), augment=augment)
        with dt[2]:
            preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)

        if batch_i < 30:
            save_img(retouch[0], batch["im_file"][0], os.path.join(save_dir, "img_results"), None, "CHW", False)
            validator.plot_val_samples(batch, batch_i)
            batch["img"] = (retouch * 255)
            validator.plot_predictions(batch, preds, batch_i)
            results_retouch = model.predict(retouch)

    validator.run_callbacks("on_val_batch_end")
    stats = validator.get_stats()
    validator.check_stats(stats)
    validator.finalize_metrics()
    validator.print_results()
    validator.run_callbacks("on_val_end")
    LOGGER.info(
        "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
            *tuple(validator.speed.values())
        )
    )

    if save_json:
        with open(str(validator.save_dir / "predictions.json"), "w") as f:
            LOGGER.info(f"Saving {f.name}...")
            json.dump(validator.jdict, f)
        stats = validator.eval_json(stats)
    if validator.args.plots or validator.args.save_json:
        LOGGER.info(f"Results saved to {colorstr('bold', validator.save_dir)}")
    metrics = validator.metrics

    if hasattr(metrics, "box") and metrics.box is not None:
        print("---- Detection (Boxes) ----")
        print(f"mAP@0.50:0.95: {metrics.box.map:.4f}")
        print(f"mAP@0.50:      {metrics.box.map50:.4f}")
        print(f"mAP@0.75:      {metrics.box.map75:.4f}")
        print("AP per class:", metrics.box.maps)

    if hasattr(metrics, "seg") and metrics.seg is not None:
        print("\n---- Segmentation (Masks) ----")
        print(f"mAP@0.50:0.95: {metrics.seg.map:.4f}")
        print(f"mAP@0.50:      {metrics.seg.map50:.4f}")
        print(f"mAP@0.75:      {metrics.seg.map75:.4f}")
        print("AP per class:", metrics.seg.maps)

    return stats

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=str(PROJECT_ROOT / 'pretrained' / 'yolo11n-seg.pt'), help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--isp_model', default='Agent', help='isp_model model')
    parser.add_argument('--isp_weights', default='experiments/xxx.pth', help='isp_weights')
    parser.add_argument('--steps', default=5, type=int, help='run step')
    parser.add_argument('--data_name', default="lis", choices=["lis"], type=str, help='data name')
    parser.add_argument('--data', type=str, default=PROJECT_ROOT / 'yolov3' / 'data' / 'lis_raw_all.yaml', help='dataset.yaml path')
    parser.add_argument('--add_noise', default=True, type=bool, help='add noise')
    parser.add_argument("--bri_range", type=float, default=None, nargs='*', help="brightness range, (low, high), 0.0~1.0")
    parser.add_argument("--noise_level", type=float, default=None, help="noise_level, 0.001~0.012")
    parser.add_argument("--save_image", action='store_true', help="save image results")
    parser.add_argument("--use_linear", action='store_true', default=False, help="use linear noise distribution")
    parser.add_argument("--pipeline", type=str, default=None, help="run with pipeline, for example: 8,3,2,5,7 ")
    parser.add_argument("--save_param", action='store_true', help="save parameter results")
    parser.add_argument("--cfg_file", type=str, default='config', help="config file")
    parser.add_argument('--project', default=PROJECT_ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='test_segmentation', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    opt.save_txt = True
    opt.save_conf = True
    if opt.pipeline is not None:
        opt.pipeline = [int(x) for x in opt.pipeline.split(",")]
        if len(opt.pipeline) < opt.steps:
            raise ValueError(f"input len(pipeline)(f{len(opt.pipeline)}) >= f{opt.steps}")
    if opt.save_param and opt.batch_size != 1:
        raise ValueError(f"If save param, input batch_size must is 1. batch-size: f{opt.batch_size}")
    return opt

def main(opt):

    if opt.task in ('train', 'val', 'test'):
        if opt.conf_thres > 0.001:
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'
        if opt.task == 'speed':
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'
                x, y = list(range(256, 1536 + 128, 128)), []
                for opt.imgsz in x:
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)
                np.savetxt(f, y, fmt='%10.4g')
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
