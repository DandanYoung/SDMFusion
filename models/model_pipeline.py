from diffusers import DDPMScheduler
from models.stable_unet_adapter import UNet2DConditionModelAdapter
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from utils.utils_ import flow_warp

def make_1step_sched(sd, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not sd or not os.path.isdir(sd):
        return None
    noise_scheduler_1step = DDPMScheduler.from_pretrained(sd, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device=device)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.to(device)
    return noise_scheduler_1step


class Conv2d(nn.Module):
    def __init__(self, in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, is_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.is_relu=is_relu
    
    def forward(self, x):
        if self.is_relu:
            return self.relu(self.conv(x))
        else:
            return self.conv(x)

class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            # ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            # ReLU(),
            Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            # ReLU(),
            Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            # ReLU(),
            Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, is_relu=False)
        )

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)]
        )

        if load_path and os.path.isfile(load_path):
            ckpt = torch.load(load_path, map_location=lambda storage, loc: storage)
            msg = self.load_state_dict(ckpt, strict=False)
            print("Loaded pretrained SpyNet weights with key remapping:", msg)
        elif load_path:
            print(f"[SPyNet] pretrained weights not found: {load_path}. Using random initialization.")

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False
                )
            )
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False
                )
            )
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow

def _pad_to_multiple(x, multiple=8, mode="reflect"):
    if x.dim() == 4:
        B, C, H, W = x.shape
        Hn = math.ceil(H / multiple) * multiple
        Wn = math.ceil(W / multiple) * multiple
        pad_h, pad_w = Hn - H, Wn - W
        if pad_h == 0 and pad_w == 0:
            return x, (H, W)
        # (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, (H, W)

    elif x.dim() == 5:
        B, T, C, H, W = x.shape
        Hn = math.ceil(H / multiple) * multiple
        Wn = math.ceil(W / multiple) * multiple
        pad_h, pad_w = Hn - H, Wn - W
        if pad_h == 0 and pad_w == 0:
            return x, (H, W)
        x_4d = x.view(B * T, C, H, W)                 # 合并(B,T)
        x_4d = F.pad(x_4d, (0, pad_w, 0, pad_h), mode=mode)
        x    = x_4d.view(B, T, C, Hn, Wn)             # 还原(B,T)
        return x, (H, W)

    else:
        raise ValueError(f"_pad_to_multiple expects 4D/5D tensor, got {x.shape} (dim={x.dim()})")

def _unpad(x, orig_hw):
    H, W = orig_hw
    return x[..., :H, :W]

def initialize_unet(pretrained_model_name_or_path=None):
    if pretrained_model_name_or_path and os.path.isdir(pretrained_model_name_or_path):
        unet = UNet2DConditionModelAdapter.from_pretrained(pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    else:
        print(
            "[UNet] pretrained model directory not found. "
            "Building a local SD2/SD-Turbo-compatible UNet skeleton for checkpoint loading."
        )
        unet = UNet2DConditionModelAdapter(cross_attention_dim=1024, use_linear_projection=True)
    unet.reset_adapter_parameters()
    conv_in = nn.Conv2d(128, 320, kernel_size=3, padding=1)
    unet.conv_in = conv_in
    conv_out = nn.Conv2d(320, 128, kernel_size=3, padding=1)
    unet.conv_out = conv_out
    return unet
    
class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.spynet = SPyNet(args.spynet_pth_root)
        self.unet = initialize_unet(args.pretrained_model_name_or_path)
        self.sched = make_1step_sched(args.pretrained_model_name_or_path, device=device)
        self.timesteps = torch.tensor([49], device=device).long()
        if self.sched is not None:
            alpha_bar_t = self.sched.alphas_cumprod[self.timesteps].view(1, 1, 1, 1)
            self.register_buffer("alpha_bar_t", alpha_bar_t, persistent=False)
        else:
            self.register_buffer("alpha_bar_t", torch.ones(1, 1, 1, 1), persistent=False)

    def one_step_update(self, z_t, residual):
        alpha_bar_t = self.alpha_bar_t.to(device=z_t.device, dtype=z_t.dtype)
        alpha_bar_t = alpha_bar_t.clamp(min=1e-8, max=1.0)
        sigma_t = (1.0 - alpha_bar_t).clamp(min=0.0).sqrt()
        return (z_t - sigma_t * residual) / alpha_bar_t.sqrt()

    def get_flow(self, frames):
        b, n, c, h, w = frames.size()
        if n <= 1:
            return frames.new_zeros(b, 0, 2, h, w)
        frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
        with torch.no_grad():
            flows_forward = self.spynet(frames_2, frames_1).view(b, n - 1, 2, h, w)
        return flows_forward

    def forward(
        self,
        feature_F_B,
        feature_F_D,
        vis_frames,
        initial_feat_prop=None,
        prev_vis_frame=None,
        return_state=False,
    ):
        base_feature_in = torch.cat((feature_F_B, feature_F_D), dim=2)
        base_feature, orig_hw = _pad_to_multiple(base_feature_in, 8, mode="reflect")
        b, t, c, h, w = base_feature.shape
        base_feature = F.interpolate(base_feature.view(b*t, c, h, w),  scale_factor=0.125,mode="bilinear", align_corners=False, antialias=True)
        base_feature = base_feature.view(b, t, c, h//8, w//8)
        
        vis_frames, _ = _pad_to_multiple(vis_frames, 8, mode="reflect")

        with torch.no_grad():
            flows_forward = self.get_flow(vis_frames)  # [B, t-1, 2, H, W]
            initial_flow = None
            if initial_feat_prop is not None and prev_vis_frame is not None:
                if prev_vis_frame.dim() == 5:
                    prev_vis_frame = prev_vis_frame[:, -1]
                prev_vis_frame, _ = _pad_to_multiple(prev_vis_frame, 8, mode="reflect")
                initial_flow = self.spynet(vis_frames[:, 0], prev_vis_frame)

        x_denoised_list = []
        for i in range(0, t):
            base_feas_i = base_feature[:, i, :, :, :]
            if i == 0:
                feat_prop_list_in = None
                if initial_feat_prop is not None:
                    feat_prop_list_in = []
                    for feat_prop in initial_feat_prop:
                        if initial_flow is not None:
                            feat_prop = flow_warp(feat_prop, initial_flow.permute(0, 2, 3, 1))
                        feat_prop_list_in.append(feat_prop)
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop_list_in = []
                for j in range(0, 4):
                    feat_prop = feat_prop_list_out[j]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  # shape[b, 4, h/8, w/8]
                    feat_prop_list_in.append(feat_prop)
            model_pred, feat_prop_list_out = self.unet(
                base_feas_i,
                timestep=self.timesteps,
                encoder_hidden_states=None,
                prior=feat_prop_list_in,
            )
            model_pred = model_pred.sample 
            x_denoised = self.one_step_update(base_feas_i, model_pred)
            x_denoised_list.append(x_denoised)
        x_denoised_out = torch.stack(x_denoised_list, dim=1)
        x_denoised_out = x_denoised_out.view(b, t, c, h//8, w//8)

        x_denoised_out = F.interpolate(x_denoised_out.view(b*t, c, h//8, w//8),  scale_factor=8, mode="bilinear", align_corners=False, antialias=True)
        
        x_hat = x_denoised_out.view(b, t, c, h, w)
        x_hat = _unpad(x_hat, orig_hw)
        flows_forward = _unpad(flows_forward, orig_hw)

    
        x_hat = x_hat + base_feature_in

        if return_state:
            return x_hat, flows_forward, feat_prop_list_out
        return x_hat, flows_forward
