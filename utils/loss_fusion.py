import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_ import flow_warp

def valid_mask_from_flow(B, H, W, device, dtype, flow_hw2):
    ones = torch.ones(B, 1, H, W, device=device, dtype=dtype)
    mask = flow_warp(ones, flow_hw2)
    return (mask > 0.999).float()

class temporal_consistency_loss(nn.Module):
    def __init__(self):
        super(temporal_consistency_loss, self).__init__()

    def forward(self, frames_fus_5d, flows_bwd):
        B, T, C, H, W = frames_fus_5d.shape
        loss_sum, count = 0.0, 0
        for t in range(1, T):
            y_t   = frames_fus_5d[:, t  , ...]   # [B,C,H,W]
            y_tm1 = frames_fus_5d[:, t-1, ...]   # [B,C,H,W]
            flow_t = flows_bwd[:, t-1, ...]
            if flow_t.dim()==4 and flow_t.shape[1]==2:
                flow_hw2 = flow_t.permute(0,2,3,1).contiguous()
            elif flow_t.dim()==4 and flow_t.shape[-1]==2:
                flow_hw2 = flow_t
            else:
                raise ValueError(f"bad flow shape at t={t}: {flow_t.shape}")
            y_tm1_warp = flow_warp(y_tm1, flow_hw2)
            mask = valid_mask_from_flow(B, H, W, y_t.device, y_t.dtype, flow_hw2)
            diff  = (y_t - y_tm1_warp) * mask
            denom = (mask.sum() * C).clamp_min(1.0)
            loss_t = diff.abs().sum() / denom
            loss_sum += loss_t
            count += 1
        return loss_sum / max(count, 1)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        # loss_grad=0.
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def cc(img1, img2, eps=1e-6):
    """Numerically stable correlation coefficient."""
    n, c = img1.shape[:2]

    img1 = img1.float().reshape(n, c, -1)
    img2 = img2.float().reshape(n, c, -1)

    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    numerator = torch.sum(img1 * img2, dim=-1)

    norm1 = torch.sqrt(torch.sum(img1.square(), dim=-1) + eps)
    norm2 = torch.sqrt(torch.sum(img2.square(), dim=-1) + eps)

    corr = numerator / (norm1 * norm2)
    corr = torch.clamp(corr, -1.0, 1.0)

    return corr.mean()