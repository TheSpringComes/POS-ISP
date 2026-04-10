
from pathlib import Path

from ultralytics import SAM, YOLO

def auto_annotate(
    data,
    det_model="yolo11x.pt",
    sam_model="sam_b.pt",
    device="",
    conf=0.25,
    iou=0.45,
    imgsz=640,
    max_det=300,
    classes=None,
    output_dir=None,
):
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
    )

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()
        if len(class_ids):
            boxes = result.boxes.xyxy
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in range(len(segments)):
                    s = segments[i]
                    if len(s) == 0:
                        continue
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
