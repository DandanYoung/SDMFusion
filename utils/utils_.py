import torch.nn.functional as F
# import pywt
import torch
from torch import nn, Tensor
from torch.autograd import Function
import kornia
from kornia import create_meshgrid
from kornia.augmentation import RandomPerspective, RandomElasticTransform
import kornia.geometry.transform as KGT
import kornia.utils as KU
import kornia.filters as KF
from kornia.filters import get_gaussian_kernel2d, filter2d
from kornia.geometry import normalize_homography, transform_points, get_perspective_transform
from kornia.utils.helpers import _torch_inverse_cast
from functools import reduce
import random
import numpy as np
from typing import Tuple
import torch.nn.functional as F
# from kornia.filters.kernels import get_gaussian_kernel2d


def set_random_seed(seed):
    # 设置Python random模块的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)
    # 设置CUDA的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #设置CuDNN的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =False

def rgb_to_ycbcr(tensor: torch.Tensor) -> torch.Tensor:
    """
    GPU-friendly YCbCr 变换，输入 tensor 范围 [0,1]
    输入: [B, 3, H, W]
    输出: [B, 3, H, W]，通道顺序 [Y, Cb, Cr]
    """
    # 分离通道
    r = tensor[:, 0:1, :, :]
    g = tensor[:, 1:2, :, :]
    b = tensor[:, 2:3, :, :]
    # 公式参考 ITU-R BT.601
    y  =  0.299    * r + 0.587    * g + 0.114    * b
    # cb = -0.168736 * r - 0.331264 * g + 0.5      * b + 0.5
    # cr =  0.5      * r - 0.418688 * g - 0.081312 * b + 0.5
    # return torch.cat([y, cb, cr], dim=2)
    return y


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    Warp an image/feature map x with optical flow.

    Args:
        x:    (N, C, H, W)
        flow: (N, Hf, Wf, 2) or (N, 2, Hf, Wf), 像素位移，右为正x，下为正y
    """
    assert x.dim() == 4, "x must be (N,C,H,W)"
    N, C, H, W = x.shape

    # 统一 flow 形状到 (N, Hf, Wf, 2)
    if flow.dim() != 4:
        raise ValueError("flow must be 4D")
    if flow.shape[1] == 2 and flow.shape[-1] != 2:
        flow = flow.permute(0, 2, 3, 1)  # (N,2,Hf,Wf) -> (N,Hf,Wf,2)
    elif flow.shape[-1] != 2:
        raise ValueError("flow must have 2 channels in the last dim")

    flow = flow.to(dtype=x.dtype)

    # 允许单个 flow 对整个 batch 进行广播
    if flow.shape[0] == 1 and N > 1:
        flow = flow.expand(N, -1, -1, -1)
    elif flow.shape[0] != N:
        raise ValueError(f"Batch size mismatch: x={N}, flow={flow.shape[0]}")

    Hf, Wf = flow.shape[1:3]

    # 如果 flow 分辨率与 x 不同：先插值到 (H,W)，再按比例缩放位移
    if (Hf, Wf) != (H, W):
        # 先把 (N,Hf,Wf,2) -> (N,2,Hf,Wf) 以便插值
        flow_chw = flow.permute(0, 3, 1, 2).contiguous()
        flow_chw = F.interpolate(flow_chw, size=(H, W), mode='bilinear', align_corners=align_corners)
        flow = flow_chw.permute(0, 2, 3, 1).contiguous()

        # 缩放位移幅度（非常关键）
        sx = float(W) / float(Wf)
        sy = float(H) / float(Hf)
        flow[..., 0] *= sx  # x 方向缩放
        flow[..., 1] *= sy  # y 方向缩放

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=x.dtype),
        torch.arange(W, device=x.device, dtype=x.dtype),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y), dim=2)  # (H,W,2)
    vgrid = base_grid.unsqueeze(0) + flow  # (N,H,W,2)

    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=-1)

    vgrid_scaled = vgrid_scaled.to(dtype=x.dtype)

    out = F.grid_sample(
        x, vgrid_scaled, mode=interp_mode,
        padding_mode=padding_mode, align_corners=align_corners
    )
    return out

def transformer(i_in: Tensor, flow: Tensor) -> [Tensor, Tensor]:
    # create mesh grid: [1, h, w, 2]
    h, w = flow.size()[1:3]
    grid = create_meshgrid(height=h, width=w, normalized_coordinates=False, device=flow.device).to(flow.dtype)
    # new locations: [b, h, w, 2]
    locs = grid + flow
    # normalize
    locs[..., 0] = (locs[..., 0] / (w - 1) - 0.5) * 2
    locs[..., 1] = (locs[..., 1] / (h - 1) - 0.5) * 2
    # apply transform
    i_out = F.grid_sample(i_in, locs, align_corners=True, mode='bilinear')
    # return moved image and flow
    return i_out, locs


def integrate(n_step: int, flow: Tensor) -> Tensor:
    scale = 1.0 / (2 ** n_step)
    flow = flow * scale
    for _ in range(n_step):
        i_flow = flow.permute(0, 3, 1, 2)  # [b, 2, h, w]
        o_flow = transformer(i_in=i_flow, flow=flow)[0].permute(0, 2, 3, 1)  # [b, h, w, 2]
        flow = flow + o_flow
    return flow


class AffineTransform(nn.Module):
    """
    Add random affine transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """
    def __init__(self, degrees=5, translate=0.1, scale=1.0, shear=None):
        super(AffineTransform, self).__init__()
        self.trs = kornia.augmentation.RandomAffine(degrees, (translate, translate), (scale, scale), shear, return_transform=True, p=1)

    def forward(self, input):
        # image shape
        batch_size, _, height, weight = input.shape
        # affine transform
        warped, affine_param = self.trs(input)  # [batch_size, 3, 3]
        affine_theta = self.param_to_theta(affine_param, weight, height)  # [batch_size, 2, 3]
        # base + disp = grid -> disp = grid - base
        base = kornia.utils.create_meshgrid(height, weight, device=input.device).to(input.dtype)
        grid = F.affine_grid(affine_theta, size=input.size(), align_corners=False)  # [batch_size, height, weight, 2]
        disp = grid - base
        return warped, -disp

    @staticmethod
    def param_to_theta(param, weight, height):
        """
        Convert affine transform matrix to theta in F.affine_grid
        :param param: affine transform matrix [batch_size, 3, 3]
        :param weight: image weight
        :param height: image height
        :return: theta in F.affine_grid [batch_size, 2, 3]
        """

        theta = torch.zeros(size=(param.shape[0], 2, 3)).to(param.device)  # [batch_size, 2, 3]

        theta[:, 0, 0] = param[:, 0, 0]
        theta[:, 0, 1] = param[:, 0, 1] * height / weight
        theta[:, 0, 2] = param[:, 0, 2] * 2 / weight + param[:, 0, 0] + param[:, 0, 1] - 1
        theta[:, 1, 0] = param[:, 1, 0] * weight / height
        theta[:, 1, 1] = param[:, 1, 1]
        theta[:, 1, 2] = param[:, 1, 2] * 2 / height + param[:, 1, 0] + param[:, 1, 1] - 1

        return theta

class ElasticTransform(nn.Module):
    """
    Add random elastic transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, kernel_size: int = 63, sigma: float = 32, alpha: Tuple[float, float]= (1.0, 1.0), align_corners: bool = False, mode: str = "bilinear"):
        super(ElasticTransform, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.alpha = alpha
        self.align_corners = align_corners
        self.mode  = mode

    def forward(self, input):
        # generate noise
        batch_size, _, height, weight = input.shape
        noise = torch.rand(batch_size, 2, height, weight) * 2 - 1  # torch.Size([16, 2, 256, 320])
        # elastic transform
        warped, disp = self.elastic_transform2d(input, noise)
        return warped, disp

    def elastic_transform2d(self, image: torch.Tensor, noise: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

        if not len(noise.shape) == 4 or noise.shape[1] != 2:
            raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

        # unpack hyper parameters
        kernel_size = self.kernel_size
        sigma = self.sigma
        alpha = self.alpha
        align_corners = self.align_corners
        mode = self.mode
        device = image.device

        # Get Gaussian kernel for 'y' and 'x' displacement
        kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
        kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x: torch.Tensor = noise[:, :1].to(device)
        disp_y: torch.Tensor = noise[:, 1:].to(device)

        disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant") * alpha[0]
        disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant") * alpha[0]

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

        # Warp image based on displacement matrix
        b, c, h, w = image.shape
        grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
        warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)
        return warped, disp




class ImgWarp(nn.Module):
    def __init__(self, level='easy'):
        super().__init__()
        if level == 'easy':
            easy = {'transforms': 'ep', 'kernel_size': (143, 143), 'sigma': (32, 32), 'distortion_scale': 0.01}
            self.adjust = RandomAdjust(easy)
        elif level == 'normal':
            normal = {'transforms': 'ep', 'kernel_size': (103, 103), 'sigma': (32, 32), 'distortion_scale': 0.3}
            self.adjust = RandomAdjust(normal)
        elif level == 'hard':
            hard = {'transforms': 'ep', 'kernel_size': (63, 63), 'sigma': (32, 32), 'distortion_scale': 0.4}
            self.adjust = RandomAdjust(hard)
        else:
            self.adjust = None

    @torch.no_grad()
    def forward(self, img_ir):
        h, w = img_ir.size()[-2:]
        grid = create_meshgrid(h, w, device=img_ir.device, dtype=img_ir.dtype)
        if self.adjust is not None:
            img_ir_w, params = self.adjust(img_ir)
            flow_gt = reduce(lambda i, j: i + j, [v for _, v in params.items()])#.to(opt.device)
            locs_gt = grid - flow_gt
        else:
            img_ir_w = img_ir
            locs_gt = grid
        return img_ir_w, locs_gt


class RandomAdjust(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # elastic
        ks, sigma = config['kernel_size'], config['sigma']
        re = RandomElasticTransform(kernel_size=ks, sigma=sigma, p=1)
        self.re = re

        # perspective
        ds = config['distortion_scale']
        rp = RandomPerspective(distortion_scale=ds, p=1)
        self.rp = rp

        # swap params
        self.size = ()
        self.device, self.dtype = torch.device('cpu'), torch.float

    def forward(self, x: Tensor) -> (Tensor, dict):
        # params
        self.size = x.size()
        self.device, self.dtype = x.device, x.dtype
        B, _, H, W = x.size()

        params = {}

        # elastic
        if 'e' in self.config['transforms']:
            x = self.re(x)
            noise = self.re._params['noise'].to(self.device)  # [b, h, w, 2]
            disp_e = self.get_elastic_disp(noise)  # [b, h, w, 2]
            # rebase
            # disp_e = disp_e.permute(0, 3, 1, 2)  # [b, 2, h, w]
            # disp_e = self.re.apply_transform(disp_e, self.re._params)
            # disp_e = disp_e.permute(0, 2, 3, 1)  # [b, h, w, 2]
            # params |= {'de': disp_e}
            params.update({'de': disp_e})

        # perspective
        if 'p' in self.config['transforms']:
            # generate params
            self.rp(x)
            # fix end_points
            corner = self.rp._params['start_points']
            self.rp._params['start_points'] = self.rp._params['end_points']
            self.rp._params['end_points'] = corner
            # transform
            x = self.rp(x, params=self.rp._params)
            # calculate offset disp
            f, t = self.rp._params['start_points'].to(x), self.rp._params['end_points'].to(x)
            matrix = get_perspective_transform(t, f)  # matrix end_points -> start_points
            disp_p = self.get_perspective_disp(matrix)  # [b, h, w, 2]
            # params |= {'dp': -disp_p}
            params.update({'dp': -disp_p})

        return x, params

    def get_perspective_disp(self, transform: Tensor) -> Tensor:
        # params
        B, _, H, W = self.size
        h_out, w_out = H, W

        # we normalize the 3x3 transformation matrix and convert to 3x4
        dst_norm_trans_src_norm = normalize_homography(transform, (H, W), (h_out, w_out))  # Bx3x3

        src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

        # this piece of code substitutes F.affine_grid since it does not support 3x3
        grid = create_meshgrid(h_out, w_out, normalized_coordinates=True, device=self.device).to(self.dtype)
        grid = grid.repeat(B, 1, 1, 1)
        disp = transform_points(src_norm_trans_dst_norm[:, None, None], grid) - grid  # disp: infrared -> \bar{infrared}
        return disp

    def get_elastic_disp(self, noise: Tensor) -> Tensor:
        # params
        config = self.config
        ks, sigma = config['kernel_size'], config['sigma']

        # Get Gaussian kernel for 'visible' and 'infrared' displacement
        kernel_x = get_gaussian_kernel2d(ks, sigma)[None]
        kernel_y = get_gaussian_kernel2d(ks, sigma)[None]
        # kernel_x = get_gaussian_kernel2d(ks, sigma)
        # kernel_y = get_gaussian_kernel2d(ks, sigma)

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x = noise[:, :1]
        disp_y = noise[:, 1:]
        disp_x = filter2d(disp_x, kernel=kernel_y, border_type="constant")
        disp_y = filter2d(disp_y, kernel=kernel_x, border_type="constant")

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)  # disp: infrared -> \bar{infrared}
        return disp



def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)

    temp = (im_flat + bias).mm(mat)

    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out