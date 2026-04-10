from types import SimpleNamespace
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from isp.denoise import NonLocalMeans, NonLocalMeansGray
from isp.sharpen import unsharp_mask, adjust_sharpness, sharpness

# =========================
# Config (from your message)
# =========================
cfg = SimpleNamespace(
    curve_steps       = 8,
    gamma_range       = 3.0,
    exposure_range    = 3.5,
    wb_range          = 1.1,
    color_curve_range = (0.90, 1.10),
    lab_curve_range   = (0.90, 1.10),   # (reserved, unused below)
    tone_curve_range  = (0.5, 2.0),
    usm_sharpen_range = (0.00001, 2.0),     # wikipedia sigma 0.5-2.0; amount 0.5-1.5 (we set defaults below)
    sharpen_range     = (0.00001, 10.0),
    ccm_range         = (-2.0, 2.0),
    denoise_range     = (0.00001, 1.0),

    masking           = False,
    minimum_strength  = 0.3,
    maximum_sharpness = 1.0,
    clamp             = False,
)

# =================
# Helper functions
# =================
def mapping(params: torch.Tensor, value_range):
    """
    Linear map from [0,1] to [l,r].
    params: (N, K, ...), entries in [0,1]
    value_range: (l, r)
    """
    l, r = value_range
    return l + (r - l) * params

def rgb2lum(image: torch.Tensor):
    """
    image: (N,3,H,W) in [0,1]
    return: (N,1,H,W)
    """
    return (0.27 * image[:, 0] + 0.67 * image[:, 1] + 0.06 * image[:, 2]).unsqueeze(1)

def lerp(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor):
    # a,b: same shape; w broadcastable (e.g., (N,1,1,1))
    return (1 - w) * a + w * b

def rgb2hsv(image: torch.Tensor):
    """
    image: (N,3,H,W) in [0,1]
    return: (N,3,H,W) in [0,1]
    """
    eps = 1e-8
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    mx, _ = torch.max(image, dim=1)   # (N,H,W)
    mn, _ = torch.min(image, dim=1)   # (N,H,W)
    diff = mx - mn

    h = torch.zeros_like(mx)
    mask = diff > eps

    # R max
    m0 = (mx == r) & mask
    h[m0] = ((g[m0] - b[m0]) / (diff[m0] + eps)) % 6
    # G max
    m1 = (mx == g) & mask
    h[m1] = ((b[m1] - r[m1]) / (diff[m1] + eps)) + 2
    # B max
    m2 = (mx == b) & mask
    h[m2] = ((r[m2] - g[m2]) / (diff[m2] + eps)) + 4

    h = (h / 6.0) % 1.0
    s = torch.zeros_like(mx)
    s[mx > eps] = diff[mx > eps] / (mx[mx > eps] + eps)
    v = mx

    return torch.stack([h, s, v], dim=1)

def hsv2rgb(hsv: torch.Tensor):
    """
    hsv: (N,3,H,W) in [0,1]
    return: (N,3,H,W) in [0,1]
    """
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)

    for k in range(6):
        m = (i == k)
        if k == 0:
            r[m], g[m], b[m] = v[m], t[m], p[m]
        elif k == 1:
            r[m], g[m], b[m] = q[m], v[m], p[m]
        elif k == 2:
            r[m], g[m], b[m] = p[m], v[m], t[m]
        elif k == 3:
            r[m], g[m], b[m] = p[m], q[m], v[m]
        elif k == 4:
            r[m], g[m], b[m] = t[m], p[m], v[m]
        elif k == 5:
            r[m], g[m], b[m] = v[m], p[m], q[m]

    return torch.stack([r, g, b], dim=1)

# ==========
# Filters
# ==========
class ContrastStretchFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'ContrastStretchFilter'
        self.range_min = (-0.05, 0.05)   
        self.range_max = (0.95, 1.05)  
        self.param_num = 2

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        min_target = mapping(params[0], self.range_min)  # (N,1) in [-r, r]
        max_target = mapping(params[1], self.range_max)  # (N,1) in [-r, r]

        min_val = img.amin(dim=(2, 3), keepdim=True)
        max_val = img.amax(dim=(2, 3), keepdim=True)

        stretched = (img - min_val) / (max_val - min_val + 1e-8) * (max_target - min_target) + min_target
        return stretched
    

class ExposureFilter(nn.Module):
    """EV exposure: multiply by 2^EV where EV in [-exposure_range, +exposure_range]."""
    def __init__(self, ev_range: float = cfg.exposure_range):
        super().__init__()
        self.name = 'ExposureFilter'
        self.range = (-ev_range, ev_range)   # EV
        self.param_num = 1

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        ev = mapping(params, self.range)  # (N,1) in [-r, r]
        return img * torch.exp(ev * math.log(2.0))
    
    def get_param(self, params):
        return mapping(params, self.range)

class GammaFilter(nn.Module):
    """Gamma correction: output = img^gamma, gamma in [1/gamma_range, gamma_range]."""
    def __init__(self, g_range: float = cfg.gamma_range):
        super().__init__()
        self.name = 'GammaFilter'
        self.range = (1.0 / g_range, g_range)
        self.param_num = 1

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        gamma = mapping(params, self.range)  # (N,1)
        return torch.pow(torch.clamp(img, 1e-6), gamma)

    def get_param(self, params):
        return mapping(params, self.range)

class GrayWhiteBalanceFilter(nn.Module):
    """
    Gray-world white balance without parameters.
    For each image in the batch:
      1) Compute per-channel means μ_R, μ_G, μ_B
      2) Target gray = (μ_R + μ_G + μ_B)/3
      3) Scale_c = target_gray / μ_c
      4) Luma normalization so 0.27*R + 0.67*G + 0.06*B stays ~constant
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.name = 'GrayWorldWhiteBalanceFilter'
        self.param_num = 3
        self.eps = eps

    @torch.no_grad()
    def forward(self, img: torch.Tensor, params: torch.Tensor):
        N, C, H, W = img.shape
        # Per-image, per-channel mean
        means = img.view(N, C, -1).mean(dim=-1)                      # (N,3)
        before = means.mean()
        target = means.mean(dim=1, keepdim=True)                      # (N,1)
        scale = target / (means + self.eps)                           # (N,3)
        # Luma-preserving normalization
        denom = (0.27 * scale[:, 0] + 0.67 * scale[:, 1] + 0.06 * scale[:, 2]).unsqueeze(1)  # (N,1)
        scale = scale / (denom + self.eps)                            # (N,3), weighted sum ~ 1
        out = img * scale.view(N, 3, 1, 1)
        after = out.view(N, C, -1).mean(dim=-1).mean()
        out = out * (before / after)
        return out


class ImprovedWhiteBalanceFilter(nn.Module):
    """
    Per-channel scale with gray-world normalization.
    scale_c in [1/wb_range, wb_range], then row-normalize by luma weights.
    """
    def __init__(self, wb_range: float = cfg.wb_range):
        super().__init__()
        self.name = 'ImprovedWhiteBalanceFilter'
        self.range = (1.0 / wb_range, wb_range)
        self.param_num = 3

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        scale = mapping(params, self.range)                  # (N,3)
        denom = 1e-6 + 0.27*scale[0] + 0.67*scale[1] + 0.06*scale[2]
        scale = scale / denom
        return img * scale.view(1, 3, 1, 1)   

    def get_param(self, params):
        return mapping(params, self.range)                  # (N,3)

class ColorFilter(nn.Module):
    def __init__(self, curve_steps: int = cfg.curve_steps, crange = cfg.color_curve_range):
        super().__init__()
        self.name = 'ColorFilter'
        self.curve_steps = int(curve_steps)
        self.range = crange
        self.param_num = 3 * self.curve_steps

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        """
        img: (N,3,H,W)
        params: (N, 3*S)
        """
        N, C, H, W = img.shape
        S = self.curve_steps
        invS = 1.0 / S

        # (N, S, 3)
        color_curve = params.view(N, S, 3)
        color_curve = mapping(color_curve, self.range)           # (N,S,3)
        curve_sum = color_curve.sum(dim=1, keepdim=True) + 1e-30 # (N,1,3)

        # img -> (N,1,3,H,W)
        x = img.unsqueeze(1)  # (N,1,3,H,W)

        # band index [0..S-1]
        device = img.device
        band_idx = torch.arange(S, device=device).view(1, S, 1, 1, 1) * invS
        # broadcast: (N,S,3,H,W)
        band = torch.clamp(x - band_idx, 0.0, invS)

        # color_curve: (N,S,3) -> (N,S,3,1,1)
        cc = color_curve.view(N, S, 3, 1, 1)

        # elementwise multiply & sum over S: (N,3,H,W)
        out = (band * cc).sum(dim=1) * (S / curve_sum.view(N,1,3,1,1))
        return out

    def get_param(self, params):
        N = params.size(0)
        S = self.curve_steps
        color_curve = params.view(N, S, 3)
        return mapping(color_curve, self.range)

# class ColorFilter(nn.Module):
#     """
#     Piecewise color curve (per channel) with cfg.curve_steps bands.
#     For each i, accumulate clamp(img - i/S, 0, 1/S) * color_curve[i].
#     """
#     def __init__(self, curve_steps: int = cfg.curve_steps, crange = cfg.color_curve_range):
#         super().__init__()
#         self.name = 'ColorFilter'
#         self.curve_steps = int(curve_steps)
#         self.range = crange          # (min, max) ~ (0.9, 1.1)
#         self.param_num = 3 * self.curve_steps

#     def forward(self, img: torch.Tensor, params: torch.Tensor):
#         S = self.curve_steps
#         color_curve = params.view(1, S, 3)                             # (N,S,3)
#         color_curve = mapping(color_curve, self.range)[:, :, :, None, None]  # (N,S,3,1,1)
#         curve_sum = torch.sum(color_curve, dim=1) + 1e-30              # (N,3,1,1)

#         out = torch.zeros_like(img)
#         invS = 1.0 / S
#         for i in range(S):
#             band = torch.clamp(img - i * invS, 0.0, invS)
#             out += band * color_curve[:, i]
#         out *= S / curve_sum
#         return out

#     def get_param(self, params):
#         S = self.curve_steps
#         color_curve = params.view(1, S, 3)                             # (N,S,3)
#         return mapping(color_curve, self.range)[:, :, :, None, None]

class ToneFilter(nn.Module):
    """
    Piecewise tone curve (shared RGB) with cfg.curve_steps bands.
    """
    def __init__(self, curve_steps: int = cfg.curve_steps, trange = cfg.tone_curve_range):
        super().__init__()
        self.name = 'ToneFilter'
        self.curve_steps = int(curve_steps)
        self.range = trange          # (0.5, 2.0)
        self.param_num = self.curve_steps

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        S = self.curve_steps
        tone_curve = mapping(params, self.range)     # (N,S,1,1,1)
        curve_sum = torch.sum(tone_curve, dim=0) + 1e-30                     # (N,1,1,1)

        out = torch.zeros_like(img)
        invS = 1.0 / S
        for i in range(S):
            band = torch.clamp(img - i * invS, 0.0, invS)
            out += band * tone_curve[i]
        out *= S / curve_sum
        return out

    def get_param(self, params):
        return mapping(params, self.range)
    
class ContrastFilter(nn.Module):
    """
    Simple contrast curve via luma shaping (cosine). 'strength' in [-1,1].
    """
    def __init__(self, strength_range: float = 1.0):
        super().__init__()
        self.name = 'ContrastFilter'
        self.range = (-strength_range, strength_range)
        self.param_num = 1

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        w = mapping(params, self.range)                   # (N,1) in [-1,1]
        lum = torch.clamp(rgb2lum(img), 0.0, 1.0)
        contrast_lum = -torch.cos(math.pi * lum) * 0.5 + 0.5  # S-curve
        contrast_img = img / (lum + 1e-6) * contrast_lum
        return lerp(img, contrast_img, w )
    
    def get_param(self, params):
        return mapping(params, self.range)  

class WNBFilter(nn.Module):
    """
    Weighted grayscale blend (black & white). alpha in [0,1].
    """
    def __init__(self):
        super().__init__()
        self.name = 'WNBFilter'
        self.range = (0.0, 1.0)
        self.param_num = 1

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        alpha = mapping(params, self.range) 
        lum = rgb2lum(img).expand_as(img)
        return lerp(img, lum, alpha)

    def get_param(self, params):
        return mapping(params, self.range) 

class SaturationPlusFilter(nn.Module):
    """
    HSV-based saturation enhancement blended with original.
    alpha in [0,1] controls blend toward enhanced saturation.
    """
    def __init__(self):
        super().__init__()
        self.name = 'SaturationPlusFilter'
        self.range = (0.0, 1.0)
        self.param_num = 1

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        img = torch.clamp(img, 0.0, 1.0)
        hsv = rgb2hsv(img)
        h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        # heuristic: raise saturation more where value is mid-range
        enhanced_s = torch.clamp(s + (1 - s) * (0.5 - torch.abs(0.5 - v)) * 0.8, 0.0, 1.0)
        full_color = hsv2rgb(torch.cat([h, enhanced_s, v], dim=1))
        alpha = mapping(params, self.range) 
        return lerp(img, full_color, alpha)

    def get_param(self, params):
        alpha = mapping(params, self.range) 
        return alpha

class DenoiseFilter(nn.Module):
    """
    Strength in [0,1]. Placeholder uses box blur; replace with your NLM/fast denoiser if available.
    """
    def __init__(self, drange = cfg.denoise_range):
        super().__init__()
        self.name = 'DenoiseFilter'
        self.range = drange
        self.param_num = 1
        self.denoise = NonLocalMeansGray(search_window_size=11, patch_size=5)  # gray mode 3x faster than rgb mode

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        strength = mapping(params, self.range) 
        return self.denoise(img, strength)

    def get_param(self, params):
        return mapping(params, self.range) 
    
class SharpenUSMFilter(nn.Module):
    """
    Unsharp mask with learnable sigma & amount.
    - sigma in [0.5, 2.0] (within cfg.usm_sharpen_range envelope)
    - amount in [0.5, 1.5] (typical values); you can change as needed.
    """
    def __init__(self):
        super().__init__()
        self.name = 'SharpenUSMFilter'
        # Keep cfg.usm_sharpen_range reserved as a global envelope if you later want to clamp
        self.range = (0.0, 2.0) 
        self.param_num = 2

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        params = mapping(params, self.range)   # (N,1)
        return unsharp_mask(img, params[:, 0], params[:, 1], kernel_size=(5, 5), clip=True)


class SharpenFilter(nn.Module):
    """
    Simple sharpening using fixed sigma (e.g., 1.0) and learnable amount in cfg.sharpen_range.
    """
    def __init__(self, amount_range = cfg.sharpen_range, fixed_sigma: float = 1.0):
        super().__init__()
        self.name = 'SharpenFilter'
        self.range = amount_range
        self.fixed_sigma = float(fixed_sigma)
        self.param_num = 1

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        amount = mapping(params, self.range)  # (N,1)
        return adjust_sharpness(img, amount )

    def get_param(self, params):
        amount = mapping(params, self.range)  # (N,1)
        return amount

class SharpenFilterV2(nn.Module):
    """
    A slightly stronger baseline than SharpenFilter by using smaller sigma (0.8).
    """
    def __init__(self, amount_range = cfg.sharpen_range, fixed_sigma: float = 0.8):
        super().__init__()
        self.name = 'SharpenFilterV2'
        self.range = amount_range
        self.fixed_sigma = float(fixed_sigma)
        self.param_num = 1

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        amount = mapping(params, self.range)  # (N,1)
        return sharpness(img, amount )

class CCMFilter(nn.Module):
    """
    3x3 color correction matrix with row normalization.
    Each element in cfg.ccm_range; rows normalized to sum 1.
    """
    def __init__(self, ccm_range = cfg.ccm_range):
        super().__init__()
        self.name = 'CCMFilter'
        self.range = ccm_range
        self.param_num = 9  # 3x3

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        ccm = mapping(params, self.range).view(1, 3, 3)      # allow negative entries
        ccm = ccm / (ccm.sum(dim=-1, keepdim=True) + 1e-8)       # row-normalize
        # Apply: out[n,h,w,:] = ccm[n] @ img[n,h,w,:]
        x = img.permute(0, 2, 3, 1)[..., None, :]  # (N,H,W,1,3)
        M = ccm[:, None, None]                     # (N,1,1,3,3)
        out = torch.sum(x * M, dim=-1)             # (N,H,W,3)
        out = out.permute(0, 3, 1, 2)
        return torch.clamp(out, 0.0, 1.0)
    
    def get_param(self, params):
        ccm = mapping(params, self.range).view(1, 3, 3)      # allow negative entries
        ccm = ccm / (ccm.sum(dim=-1, keepdim=True) + 1e-8)       # row-normalize
        return torch.flatten(ccm)

  
# Optionally expose all classes
__all__ = [
    'ExposureFilter', 'GammaFilter', 'ImprovedWhiteBalanceFilter', 'ContrastStretchFilter',
    'ColorFilter', 'ToneFilter', 'ContrastFilter',
    'WNBFilter', 'SaturationPlusFilter', 'DenoiseFilter',
    'SharpenUSMFilter', 'SharpenFilter', 'SharpenFilterV2',
    'CCMFilter', 'GrayWhiteBalanceFilter',
    # helpers
    'mapping', 'rgb2lum', 'rgb2hsv', 'hsv2rgb',
]
