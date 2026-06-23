import argparse
import datetime
import os
import random
import time
import warnings

import kornia
import torch
import torch.nn as nn
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from models.model_pipeline import Generator
from models.model_stage1 import DE_Decoder, DE_Encoder, HighFreqExtractor, LowFreqExtractor
from utils.loss_fusion import Fusionloss, cc, temporal_consistency_loss
from utils.utils_ import set_random_seed


warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SDMFusion training entry.")
    parser.add_argument("--mode", choices=["stage1", "stage2", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--output_dir", default="outputs")

    parser.add_argument("--train_root_stage1", default=None)
    parser.add_argument("--train_root_stage2", default=None)
    parser.add_argument("--train_root", default="datasets/train")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--max_train_batches", type=int, default=None)

    parser.add_argument("--stage1_epochs", type=int, default=1200)
    parser.add_argument("--stage1_phase_gap", type=int, default=400)
    parser.add_argument("--stage1_batch_size", type=int, default=12)
    parser.add_argument("--stage1_seq_length", type=int, default=1)
    parser.add_argument("--stage1_crop_size", type=int, default=128)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage1_min_lr", type=float, default=1e-6)
    parser.add_argument("--stage1_weight_decay", type=float, default=0.0)
    parser.add_argument("--stage1_optim_step", type=int, default=20)
    parser.add_argument("--stage1_optim_gamma", type=float, default=0.5)
    parser.add_argument("--stage1_clip_grad_norm", type=float, default=0.01)

    parser.add_argument("--stage2_epochs", type=int, default=500)
    parser.add_argument("--stage2_batch_size", type=int, default=1)
    parser.add_argument("--stage2_seq_length", type=int, default=6)
    parser.add_argument("--stage2_crop_size", type=int, default=320)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_weight_decay", type=float, default=1e-4)
    parser.add_argument("--stage2_clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--stage2_train_decoder", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--pretrained_model_name_or_path", default="pretrained/sd-turbo")
    parser.add_argument("--spynet_pth_root", default="pretrained/spynet.pth")
    parser.add_argument("--stage1_ckpt_path", default=None)

    parser.add_argument("--coeff_mse_loss_VF", type=float, default=1.0)
    parser.add_argument("--coeff_mse_loss_IF", type=float, default=1.0)
    parser.add_argument("--coeff_decomp", type=float, default=2.0)
    parser.add_argument("--coeff_tv", type=float, default=5.0)
    parser.add_argument("--coeff_tc", type=float, default=0.1)
    return parser.parse_args()


class RandomCropWithPosition(T.RandomCrop):
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img):
        width, height = img.size
        crop_height, crop_width = self.size
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        return img.crop((left, top, left + crop_width, top + crop_height)), top, left


class FusionTrainDataset(Dataset):
    def __init__(self, root_dir, seq_length=1, crop_size=128, frame_step=2):
        super().__init__()
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.frame_step = frame_step
        self.video_list = natsorted(
            name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))
        )
        self.transform = RandomCropWithPosition(crop_size)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.video_list)

    @staticmethod
    def _image_files(path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return natsorted(
            name for name in os.listdir(path) if os.path.splitext(name.lower())[1] in exts
        )

    @staticmethod
    def collate_fn(batch):
        images_a, images_b, top, left, name = zip(*batch)
        return torch.stack(images_a, dim=0), torch.stack(images_b, dim=0), top, left, name

    def _resolve_dirs(self, video_path):
        if os.path.isdir(os.path.join(video_path, "channel")):
            return "channel", "channel2", "not156"
        if os.path.isdir(os.path.join(video_path, "visible")):
            return "visible", "infrared", "generic"
        raise FileNotFoundError(
            f"{video_path} must contain either channel/channel2 or visible/infrared."
        )

    def __getitem__(self, index):
        video_name = self.video_list[index]
        video_path = os.path.join(self.root_dir, video_name)
        vis_dir, ir_dir, fmt = self._resolve_dirs(video_path)
        frame_list = self._image_files(os.path.join(video_path, vis_dir))

        need = self.frame_step * self.seq_length
        if len(frame_list) < need:
            raise RuntimeError(
                f"{video_name} has {len(frame_list)} frames, but needs at least {need}."
            )
        start_idx = random.randint(0, len(frame_list) - need)

        vis_frames, ir_frames = [], []
        top = left = 0
        last_frame_name = frame_list[start_idx]
        for i in range(start_idx, start_idx + need, self.frame_step):
            vis_frame_name = frame_list[i]
            ir_frame_name = vis_frame_name.replace("ll", "t") if fmt == "not156" else vis_frame_name
            vis_frame = Image.open(os.path.join(video_path, vis_dir, vis_frame_name)).convert("L")
            ir_frame = Image.open(os.path.join(video_path, ir_dir, ir_frame_name)).convert("L")

            vis_cropped, top, left = self.transform(vis_frame)
            crop_w, crop_h = vis_cropped.size
            ir_cropped = ir_frame.crop((left, top, left + crop_w, top + crop_h))

            vis_frames.append(self.to_tensor(vis_cropped))
            ir_frames.append(self.to_tensor(ir_cropped))
            last_frame_name = vis_frame_name

        return torch.stack(vis_frames), torch.stack(ir_frames), top, left, last_frame_name


def resolve_device(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(args.device)


def ensure_dirs(args):
    os.makedirs(args.output_dir, exist_ok=True)


def build_stage1_models(device):
    return (
        DE_Encoder().to(device),
        DE_Decoder().to(device),
        LowFreqExtractor(dim=64).to(device),
        HighFreqExtractor(num_layers=3).to(device),
    )


def load_strip_module(model, state, strict=True):
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)


GENERATOR_DELTA_PREFIXES = (
    "unet.conv_in.",
    "unet.conv_out.",
    "unet.up_adapter_blocks.",
)


def generator_delta_state_dict(generator):
    """Return only Stage II parameters that are not restored from pretrained files."""
    return {
        name: tensor
        for name, tensor in generator.state_dict().items()
        if name.startswith(GENERATOR_DELTA_PREFIXES)
    }


def save_checkpoint(path, epoch, args, models, optimizer=None, scheduler=None, best_loss=None):
    checkpoint = {
        "epoch": epoch,
        "args": vars(args),
        "DIDF_Encoder": models["encoder"].state_dict(),
        "DIDF_Decoder": models["decoder"].state_dict(),
        "BaseFuseLayer": models["base"].state_dict(),
        "DetailFuseLayer": models["detail"].state_dict(),
        "best_loss": best_loss,
    }
    if models.get("generator") is not None:
        checkpoint["format"] = "sdmfusion-stage2-delta-v1"
        checkpoint["gen_model_delta"] = generator_delta_state_dict(models["generator"])
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def set_trainable_in_generator(gen):
    def num_params(module, trainable_only=False):
        if module is None:
            return 0
        params = module.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def freeze(module):
        if module is None:
            return
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def unfreeze(module):
        if module is None:
            return []
        for p in module.parameters():
            p.requires_grad = True
        return list(module.parameters())

    for p in gen.parameters():
        p.requires_grad = False

    freeze(getattr(gen, "spynet", None))
    unet = getattr(gen, "unet", None)
    if unet is None:
        raise AttributeError("Generator is missing attribute `unet`.")
    freeze(unet)

    train_params = []
    conv_in = getattr(unet, "conv_in", None)
    conv_out = getattr(unet, "conv_out", None)
    up_adapters = getattr(unet, "up_adapter_blocks", None)

    train_params += unfreeze(conv_in)
    train_params += unfreeze(conv_out)
    if up_adapters is not None:
        for block in up_adapters:
            train_params += unfreeze(block)

    fmt = lambda n: f"{n / 1e6:.3f} M"
    print(f"[Stage2] Generator total params: {fmt(num_params(gen))}")
    print(f"[Stage2] Generator trainable params: {fmt(num_params(gen, True))}")
    print(f"[Stage2] Selected trainable params: {fmt(sum(p.numel() for p in train_params))}")
    return train_params


def build_loader(root, seq_length, crop_size, batch_size, frame_step, num_workers, shuffle=True):
    dataset = FusionTrainDataset(
        root_dir=root,
        seq_length=seq_length,
        crop_size=crop_size,
        frame_step=frame_step,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        collate_fn=FusionTrainDataset.collate_fn,
    )


def train_stage1(args, device):
    train_root = args.train_root_stage1 or args.train_root
    loader = build_loader(
        train_root,
        args.stage1_seq_length,
        args.stage1_crop_size,
        args.stage1_batch_size,
        args.frame_step,
        args.num_workers,
        shuffle=False,
    )
    encoder, decoder, base_fuse, detail_fuse = build_stage1_models(device)
    criteria_fusion = Fusionloss()
    mse_loss_fn = nn.MSELoss()
    l1_loss_fn = nn.L1Loss()

    optimizers = [
        torch.optim.Adam(encoder.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay),
        torch.optim.Adam(decoder.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay),
        torch.optim.Adam(base_fuse.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay),
        torch.optim.Adam(detail_fuse.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay),
    ]
    schedulers = [
        torch.optim.lr_scheduler.StepLR(
            opt, step_size=args.stage1_optim_step, gamma=args.stage1_optim_gamma
        )
        for opt in optimizers
    ]

    checkpoint_dir = args.output_dir
    stale_best_path = os.path.join(checkpoint_dir, "stage1_best.pth")
    last_path = os.path.join(checkpoint_dir, "stage1_last.pth")
    if os.path.exists(stale_best_path):
        os.remove(stale_best_path)
    prev_time = time.time()

    print(f"[Stage1] training on {train_root}")
    for epoch in range(args.stage1_epochs):
        epoch_losses = []
        for batch_idx, (data_vis, data_ir, _, _, _) in enumerate(tqdm(loader, desc=f"Stage1 {epoch + 1}/{args.stage1_epochs}")):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            data_vis = data_vis.to(device, non_blocking=True)
            data_ir = data_ir.to(device, non_blocking=True)
            b, t, _, h, w = data_ir.size()
            data_vis = data_vis.reshape(b * t, 1, h, w)
            data_ir = data_ir.reshape(b * t, 1, h, w)

            encoder.train()
            decoder.train()
            base_fuse.train()
            detail_fuse.train()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if epoch < args.stage1_phase_gap:
                feature_v_b, feature_v_d, _ = encoder(data_vis)
                feature_i_b, feature_i_d, _ = encoder(data_ir)
                data_vis_hat, _ = decoder([data_vis], feature_v_b, feature_v_d)
                data_ir_hat, _ = decoder([data_ir], feature_i_b, feature_i_d)

                cc_loss_b = cc(feature_v_b, feature_i_b)
                cc_loss_d = cc(feature_v_d, feature_i_d)
                mse_loss_v = 5 * kornia.losses.ssim_loss(data_vis, data_vis_hat, window_size=11, reduction="mean") + mse_loss_fn(data_vis, data_vis_hat)
                mse_loss_i = 5 * kornia.losses.ssim_loss(data_ir, data_ir_hat, window_size=11, reduction="mean") + mse_loss_fn(data_ir, data_ir_hat)
                fusionloss = l1_loss_fn(kornia.filters.SpatialGradient()(data_vis), kornia.filters.SpatialGradient()(data_vis_hat))
                loss_decomp = (cc_loss_d) ** 2 / (1.01 + cc_loss_b)
                loss = (
                    args.coeff_mse_loss_VF * mse_loss_v
                    + args.coeff_mse_loss_IF * mse_loss_i
                    + args.coeff_decomp * loss_decomp
                    + args.coeff_tv * fusionloss
                )
                active_models = [encoder, decoder]
                active_optimizers = optimizers[:2]
            else:
                feature_v_b, feature_v_d, _ = encoder(data_vis)
                feature_i_b, feature_i_d, _ = encoder(data_ir)
                feature_f_b = base_fuse(feature_i_b + feature_v_b)
                feature_f_d = detail_fuse(feature_i_d + feature_v_d)
                data_fuse, _ = decoder([data_vis, data_ir], feature_f_b, feature_f_d)

                mse_loss_v = 2 * kornia.losses.ssim_loss(data_vis, data_fuse, window_size=11, reduction="mean")
                mse_loss_i = 2 * kornia.losses.ssim_loss(data_ir, data_fuse, window_size=11, reduction="mean")
                cc_loss_b = cc(feature_v_b, feature_i_b)
                cc_loss_d = cc(feature_v_d, feature_i_d)
                loss_decomp = (cc_loss_d) ** 2 / (1.01 + cc_loss_b)
                total_fusion_loss, _, _ = criteria_fusion(data_vis, data_ir, data_fuse)
                loss = (
                    args.coeff_decomp * loss_decomp
                    + total_fusion_loss
                    + args.coeff_mse_loss_VF * mse_loss_v
                    + args.coeff_mse_loss_IF * mse_loss_i
                )
                active_models = [encoder, decoder, base_fuse, detail_fuse]
                active_optimizers = optimizers

            loss.backward()
            for model in active_models:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.stage1_clip_grad_norm, norm_type=2
                )
            for opt in active_optimizers:
                opt.step()
            epoch_losses.append(loss.detach().item())

        for scheduler in schedulers[:2]:
            scheduler.step()
        if not epoch < args.stage1_phase_gap:
            for scheduler in schedulers[2:]:
                scheduler.step()
        for optimizer in optimizers:
            if optimizer.param_groups[0]["lr"] <= args.stage1_min_lr:
                optimizer.param_groups[0]["lr"] = args.stage1_min_lr

        epoch_loss = float(torch.tensor(epoch_losses).mean().item())
        models = {"encoder": encoder, "decoder": decoder, "base": base_fuse, "detail": detail_fuse}
        save_checkpoint(last_path, epoch + 1, args, models, best_loss=epoch_loss)
       
       

        elapsed = datetime.timedelta(seconds=int(time.time() - prev_time))
        prev_time = time.time()
        # print(f"[Stage1] epoch={epoch + 1} loss={epoch_loss:.6f} best={best_loss:.6f} elapsed={elapsed}")
        print(f"[Stage1] epoch={epoch + 1} loss={epoch_loss:.6f} elapsed={elapsed}")

    return last_path


def train_stage2(args, device, stage1_ckpt_path):
    if not stage1_ckpt_path:
        raise ValueError("--stage1_ckpt_path is required when --mode stage2.")
    train_root = args.train_root_stage2 or args.train_root
    loader = build_loader(
        train_root,
        args.stage2_seq_length,
        args.stage2_crop_size,
        args.stage2_batch_size,
        args.frame_step,
        args.num_workers,
    )

    encoder, decoder, base_fuse, detail_fuse = build_stage1_models(device)
    ckpt = torch.load(stage1_ckpt_path, map_location=device)
    load_strip_module(encoder, ckpt["DIDF_Encoder"])
    load_strip_module(decoder, ckpt["DIDF_Decoder"])
    load_strip_module(base_fuse, ckpt["BaseFuseLayer"])
    load_strip_module(detail_fuse, ckpt["DetailFuseLayer"])

    for module in [encoder, base_fuse, detail_fuse]:
        module.eval()
        for p in module.parameters():
            p.requires_grad = False
    decoder.train(args.stage2_train_decoder)
    for p in decoder.parameters():
        p.requires_grad = args.stage2_train_decoder

    args.device = str(device)
    gen_model = Generator(args).to(device)
    optim_params = set_trainable_in_generator(gen_model)
    if args.stage2_train_decoder:
        optim_params += list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        optim_params, lr=args.stage2_lr, weight_decay=args.stage2_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.stage2_epochs * len(loader)),
        eta_min=args.stage2_lr * 0.1,
    )

    criteria_fusion = Fusionloss()
    temporal_consistency = temporal_consistency_loss()
    mse_loss_fn = nn.MSELoss()
    best_loss = float("inf")
    checkpoint_dir = args.output_dir
    best_path = os.path.join(checkpoint_dir, "stage2.pt")
    last_path = os.path.join(checkpoint_dir, "stage2_last.pt")
    for name in os.listdir(checkpoint_dir):
        if name.startswith("stage2_epoch_") and name.endswith((".pth", ".pt")):
            os.remove(os.path.join(checkpoint_dir, name))

    print(f"[Stage2] training on {train_root}")
    print(f"[Stage2] loading stage1 checkpoint: {stage1_ckpt_path}")
    for epoch in range(args.stage2_epochs):
        epoch_losses, epoch_tc_losses = [], []
        gen_model.train()
        decoder.train(args.stage2_train_decoder)
        for batch_idx, (data_vis, data_ir, _, _, _) in enumerate(tqdm(loader, desc=f"Stage2 {epoch + 1}/{args.stage2_epochs}")):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            data_vis = data_vis.to(device, non_blocking=True)
            data_ir = data_ir.to(device, non_blocking=True)
            b, t, c, h, w = data_ir.shape
            data_vis_in = data_vis.view(b * t, c, h, w)
            data_ir_in = data_ir.view(b * t, c, h, w)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                feature_v_b, feature_v_d, _ = encoder(data_vis_in)
                feature_i_b, feature_i_d, _ = encoder(data_ir_in)
                feature_f_b = base_fuse(feature_i_b + feature_v_b)
                feature_f_d = detail_fuse(feature_i_d + feature_v_d)

            feature_f_b_5d = feature_f_b.view(b, t, -1, h, w)
            feature_f_d_5d = feature_f_d.view(b, t, -1, h, w)
            fea_gen, flows_forward = gen_model(feature_f_b_5d, feature_f_d_5d, data_vis)

            detail_pair = torch.cat((feature_v_d, feature_i_d), dim=1).view(b, t, -1, h, w)
            fea_gen = (fea_gen + detail_pair).view(b * t, -1, h, w)
            data_fuse, _ = decoder(
                data_vis_in * 0.5 + data_ir_in * 0.5,
                fea_gen,
                detail_feature=None,
                is_2dfusion=True,
            )

            data_fuse_5d = data_fuse.view(b, t, -1, h, w)
            tc_loss = temporal_consistency(data_fuse_5d, flows_forward)
            if not torch.is_tensor(tc_loss):
                tc_loss = data_fuse.new_tensor(tc_loss)
            mse_loss_v = 5 * kornia.losses.ssim_loss(data_vis_in, data_fuse, window_size=11, reduction="mean") + mse_loss_fn(data_vis_in, data_fuse)
            mse_loss_i = 5 * kornia.losses.ssim_loss(data_ir_in, data_fuse, window_size=11, reduction="mean") + mse_loss_fn(data_ir_in, data_fuse)
            fusionloss, _, _ = criteria_fusion(data_vis_in, data_ir_in, data_fuse)
            loss = (
                args.coeff_mse_loss_VF * mse_loss_v
                + args.coeff_mse_loss_IF * mse_loss_i
                + args.coeff_tv * fusionloss
                + args.coeff_tc * tc_loss
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optim_params,
                max_norm=args.stage2_clip_grad_norm,
                norm_type=2,
                error_if_nonfinite=False,
            )

            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.detach().item())
            epoch_tc_losses.append(tc_loss.detach().item())

        if epoch_losses:
            epoch_loss = float(torch.tensor(epoch_losses).mean().item())
            epoch_tc_loss = float(torch.tensor(epoch_tc_losses).mean().item())
        else:
            epoch_loss = float("inf")
            epoch_tc_loss = float("inf")
        models = {
            "encoder": encoder,
            "decoder": decoder,
            "base": base_fuse,
            "detail": detail_fuse,
            "generator": gen_model,
        }
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
        save_checkpoint(last_path, epoch + 1, args, models, best_loss=best_loss)
        if is_best:
            save_checkpoint(best_path, epoch + 1, args, models, best_loss=best_loss)
        print(
            f"[Stage2] epoch={epoch + 1} loss={epoch_loss:.6f} "
            f"tc={epoch_tc_loss:.6f} best={best_loss:.6f}"
        )

    return best_path


def main():
    args = parse_args()
    ensure_dirs(args)
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args)
    args.device = str(device)

    start_time = time.time()
    stage1_ckpt = args.stage1_ckpt_path
    if args.mode in {"stage1", "all"}:
        stage1_ckpt = train_stage1(args, device)
    if args.mode in {"stage2", "all"}:
        train_stage2(args, device, stage1_ckpt)
    hours = (time.time() - start_time) / 3600
    print(f"[Done] mode={args.mode}, total training time={hours:.4f} hours")


if __name__ == "__main__":
    main()
