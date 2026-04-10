
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

import cv2

from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    DEFAULT_SOL_DICT,
    IS_VSCODE,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_FILE,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    checks,
    colorstr,
    deprecation_warn,
    vscode_msg,
    yaml_load,
    yaml_print,
)

SOLUTION_MAP = {
    "count": ("ObjectCounter", "count"),
    "heatmap": ("Heatmap", "generate_heatmap"),
    "queue": ("QueueManager", "process_queue"),
    "speed": ("SpeedEstimator", "estimate_speed"),
    "workout": ("AIGym", "monitor"),
    "analytics": ("Analytics", "process_data"),
    "trackzone": ("TrackZone", "trackzone"),
    "inference": ("Inference", "inference"),
    "help": None,
}

MODES = {"train", "val", "predict", "export", "track", "benchmark"}
TASKS = {"detect", "segment", "classify", "pose", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "classify": "yolo11n-cls.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}

ARGV = sys.argv or ["", ""]
SOLUTIONS_HELP_MSG = f"""
    Arguments received: {str(["yolo"] + ARGV[1:])}. Ultralytics 'yolo solutions' usage overview:

        yolo solutions SOLUTION ARGS

        Where SOLUTION (optional) is one of {list(SOLUTION_MAP.keys())[:-1]}
              ARGS (optional) are any number of custom 'arg=value' pairs like 'show_in=True' that override defaults
                  at https://docs.ultralytics.com/usage/cfg

    1. Call object counting solution
        yolo solutions count source="path/to/video/file.mp4" region=[(20, 400), (1080, 400), (1080, 360), (20, 360)]

    2. Call heatmaps solution
        yolo solutions heatmap colormap=cv2.COLORMAP_PARULA model=yolo11n.pt

    3. Call queue management solution
        yolo solutions queue region=[(20, 400), (1080, 400), (1080, 360), (20, 360)] model=yolo11n.pt

    4. Call workouts monitoring solution for push-ups
        yolo solutions workout model=yolo11n-pose.pt kpts=[6, 8, 10]

    5. Generate analytical graphs
        yolo solutions analytics analytics_type="pie"

    6. Track objects within specific zones
        yolo solutions trackzone source="path/to/video/file.mp4" region=[(150, 150), (1130, 150), (1130, 570), (150, 570)]

    7. Streamlit real-time webcam inference GUI
        yolo streamlit-predict
    """
CLI_HELP_MSG = f"""
    Arguments received: {str(["yolo"] + ARGV[1:])}. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {TASKS}
                MODE (required) is one of {MODES}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

    5. Ultralytics solutions usage
        yolo solutions count or in {list(SOLUTION_MAP.keys())[1:-1]} source="path/to/video/file.mp4"

    6. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        yolo solutions help

    Docs: https://docs.ultralytics.com
    Solutions: https://docs.ultralytics.com/solutions/
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

CFG_FLOAT_KEYS = {
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
}
CFG_FRACTION_KEYS = {
    "dropout",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}
CFG_INT_KEYS = {
    "epochs",
    "patience",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}

def cfg2dict(cfg):
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg

def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    cfg = cfg2dict(cfg)

    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}

    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":
        cfg["name"] = str(cfg.get("model", "")).split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    check_cfg(cfg)

    return IterableSimpleNamespace(**cfg)

def check_cfg(cfg, hard=True):
    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)

def get_save_dir(args, name=None):
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir)

def _handle_deprecation(custom):
    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")
        if key == "label_smoothing":
            deprecation_warn(key)
            custom.pop("label_smoothing")

    return custom

def check_dict_alignment(base: Dict, custom: Dict, e=None):
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    if mismatched := [k for k in custom_keys if k not in base_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e

def merge_equals_args(args: List[str]) -> List[str]:
    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "=" and 0 < i < len(args) - 1:
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:
            new_args[-1] += arg
            i += 1
            continue

        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    if current:
        new_args.append(current)

    return new_args

def handle_yolo_hub(args: List[str]) -> None:
    from ultralytics import hub

    if args[0] == "login":
        key = args[1] if len(args) > 1 else ""

        hub.login(key)
    elif args[0] == "logout":

        hub.logout()

def handle_yolo_settings(args: List[str]) -> None:
    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()
                SETTINGS.reset()
                LOGGER.info("Settings reset successfully")
            else:
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)

        print(SETTINGS)
        LOGGER.info(f"💡 Learn more about Ultralytics Settings at {url}")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ settings error: '{e}'. Please see {url} for help.")

def handle_yolo_solutions(args: List[str]) -> None:
    full_args_dict = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}
    overrides = {}

    for arg in merge_equals_args(args):
        arg = arg.lstrip("-").rstrip(",")
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ""}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)

    if args and args[0] in SOLUTION_MAP:
        if args[0] != "help":
            s_n = args.pop(0)
        else:
            LOGGER.info(SOLUTIONS_HELP_MSG)
    else:
        LOGGER.warning(
            f"⚠️ No valid solution provided. Using default 'count'. Available: {', '.join(SOLUTION_MAP.keys())}"
        )
        s_n = "count"

    if args and args[0] == "help":
        return

    if s_n == "inference":
        checks.check_requirements("streamlit>=1.29.0")
        LOGGER.info("💡 Loading Ultralytics live inference app...")
        subprocess.run(
            [
                "streamlit",
                "run",
                str(ROOT / "solutions/streamlit_inference.py"),
                "--server.headless",
                "true",
                overrides.pop("model", "yolo11n.pt"),
            ]
        )
    else:
        cls, method = SOLUTION_MAP[s_n]

        from ultralytics import solutions

        solution = getattr(solutions, cls)(IS_CLI=True, **overrides)
        process = getattr(
            solution, method
        )

        cap = cv2.VideoCapture(solution.CFG["source"])

        import os
        from pathlib import Path

        from ultralytics.utils.files import increment_path

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        if s_n == "analytics":
            w, h = 1920, 1080
        save_dir = increment_path(Path("runs") / "solutions" / "exp", exist_ok=False)
        save_dir.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(os.path.join(save_dir, "solution.avi"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        try:
            f_n = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = process(frame, f_n := f_n + 1) if s_n == "analytics" else process(frame)
                vw.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()

def parse_key_value_pair(pair: str = "key=value"):
    k, v = pair.split("=", 1)
    k, v = k.strip(), v.strip()
    assert v, f"missing '{k}' value"
    return k, smart_value(v)

def smart_value(v):
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return eval(v)
        except Exception:
            return v

def entrypoint(debug=""):
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: yaml_print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "logout": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    special.update({k[0]: v for k, v in special.items()})
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}
    for a in merge_equals_args(args):
        if a.startswith("--"):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    check_dict_alignment(full_args_dict, overrides)

    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"WARNING ⚠️ 'mode' argument is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")

    task = overrides.pop("task", None)
    if task:
        if task == "classify" and mode == "track":
            raise ValueError(
                f"❌ Classification doesn't support 'mode=track'. Valid modes for classification are"
                f" {MODES - {'track'}}.\n{CLI_HELP_MSG}"
            )
        elif task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    "WARNING ⚠️ invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo11n.pt"
        LOGGER.warning(f"WARNING ⚠️ 'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:
        from ultralytics import RTDETR

        model = RTDETR(model)
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
    if isinstance(overrides.get("pretrained"), str):
        model.load(overrides["pretrained"])

    if task != model.task:
        if task:
            LOGGER.warning(
                f"WARNING ⚠️ conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"WARNING ⚠️ 'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"WARNING ⚠️ 'format' argument is missing. Using default 'format={overrides['format']}'.")

    getattr(model, mode)(**overrides)

    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")

    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())

def copy_default_cfg():
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )

if __name__ == "__main__":

    entrypoint(debug="")
