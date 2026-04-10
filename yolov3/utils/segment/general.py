import cv2
import numpy as np
import torch
import torch.nn.functional as F

def crop_mask(masks, boxes):

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask_upsample(protos, masks_in, bboxes, shape):

    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.5)

def process_mask(protos, masks_in, bboxes, shape, upsample=False):

    c, mh, mw = protos.shape
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]
    return masks.gt_(0.5)

def process_mask_native(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    gain = min(mh / shape[0], mw / shape[1])
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2
    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.5)

def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]

    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

def mask_iou(mask1, mask2, eps=1e-7):
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection
    return intersection / (union + eps)

def masks_iou(mask1, mask2, eps=1e-7):
    intersection = (mask1 * mask2).sum(1).clamp(0)
    union = (mask1.sum(1) + mask2.sum(1))[None] - intersection
    return intersection / (union + eps)

def masks2segments(masks, strategy='largest'):

    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))
        segments.append(c.astype('float32'))
    return segments
