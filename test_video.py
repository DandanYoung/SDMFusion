import argparse
import json
import os
import time
import warnings
from glob import glob


warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DATASET_CHOICES = ["hdo", "m3svd", "not156", "vtmot"]


def parse_args():
    parser = argparse.ArgumentParser(description="Open-source Stage2 inference for SDMFusion.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pretrained_model_name_or_path", default="pretrained/sd-turbo")
    parser.add_argument("--spynet_pth_root", default="pretrained/spynet.pth")
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--add_input_offset", type=float, default=0.1)
    parser.add_argument("--no_detail_residual", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_stream_state", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join("results", args.dataset.lower())
    return args


def resolve_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_strip_module(model, state, strict=True):
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)


def build_dataset(args):
    import torch
    from natsort import natsorted
    from PIL import Image
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms

    from datasets.dataset_test_data import HDO, M3SVD, NOT156, VTMOT

    class FlexiblePairDataset(Dataset):
        def __init__(self, root_dir, seq_len=1, overlap=0):
            self.seq_len = seq_len
            self.step = seq_len - overlap
            if self.step <= 0:
                raise ValueError("--overlap must be smaller than --seq_len.")
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.index_list = []
            for video_name in natsorted(os.listdir(root_dir)):
                video_dir = os.path.join(root_dir, video_name)
                if not os.path.isdir(video_dir):
                    continue
                if os.path.isdir(os.path.join(video_dir, "channel")):
                    vis_dir, ir_dir, pattern = "channel", "channel2", "*.jpg"
                elif os.path.isdir(os.path.join(video_dir, "visible")):
                    vis_dir, ir_dir, pattern = "visible", "infrared", "*.*"
                else:
                    continue
                vis_paths = natsorted(glob(os.path.join(video_dir, vis_dir, pattern)))
                ir_paths = natsorted(glob(os.path.join(video_dir, ir_dir, pattern)))
                if not vis_paths or len(vis_paths) != len(ir_paths):
                    raise RuntimeError(
                        f"{video_name}: visible/infrared frame counts differ "
                        f"({len(vis_paths)} vs {len(ir_paths)})."
                    )
                starts = list(range(0, len(vis_paths) - seq_len + 1, self.step))
                last_start = len(vis_paths) - seq_len
                if last_start < 0:
                    continue
                if not starts or starts[-1] != last_start:
                    starts.append(last_start)
                for start in starts:
                    self.index_list.append((video_name, vis_paths, ir_paths, start))

        def __len__(self):
            return len(self.index_list)

        def __getitem__(self, idx):
            video_name, vis_paths, ir_paths, start = self.index_list[idx]
            vis_seq, ir_seq, frame_names = [], [], []
            for i in range(start, start + self.seq_len):
                vis_path = vis_paths[i]
                ir_path = ir_paths[i]
                frame_names.append(os.path.basename(vis_path).replace("ll", ""))
                vis = Image.open(vis_path).convert("L")
                ir = Image.open(ir_path).convert("L")
                vis_seq.append(self.transform(vis))
                ir_seq.append(self.transform(ir))
            return video_name, frame_names, torch.stack(vis_seq, dim=0), torch.stack(ir_seq, dim=0)

    dataset_registry = {
        "not156": NOT156,
        "hdo": HDO,
        "m3svd": M3SVD,
        "vtmot": VTMOT,
    }
    direct_roots = {
        "not156": os.path.join(args.data_root, "NOT156"),
        "vtmot": os.path.join(args.data_root, "VTMOT"),
    }
    direct_root = direct_roots.get(args.dataset)
    if direct_root and os.path.isdir(direct_root):
        return FlexiblePairDataset(direct_root, seq_len=args.seq_len, overlap=args.overlap)
    dataset_cls = dataset_registry[args.dataset]
    return dataset_cls(args.data_root, seq_len=args.seq_len, overlap=args.overlap)


def build_models(args, device):
    from models.model_pipeline import Generator
    from models.model_stage1 import DE_Decoder, DE_Encoder, HighFreqExtractor, LowFreqExtractor

    encoder = DE_Encoder().to(device)
    decoder = DE_Decoder().to(device)
    base = LowFreqExtractor(dim=64).to(device)
    detail = HighFreqExtractor(num_layers=3).to(device)
    args.device = str(device)
    generator = Generator(args).to(device)
    return encoder, decoder, base, detail, generator


def load_checkpoint(ckpt_path, models, device):
    required = ["DIDF_Encoder", "DIDF_Decoder", "BaseFuseLayer", "DetailFuseLayer"]
    ckpt = torch.load(ckpt_path, map_location=device)
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")

    encoder, decoder, base, detail, generator = models
    load_strip_module(encoder, ckpt["DIDF_Encoder"])
    load_strip_module(decoder, ckpt["DIDF_Decoder"])
    load_strip_module(base, ckpt["BaseFuseLayer"])
    load_strip_module(detail, ckpt["DetailFuseLayer"])
    if "gen_model_delta" in ckpt:
        incompatible = generator.load_state_dict(ckpt["gen_model_delta"], strict=False)
        if incompatible.unexpected_keys:
            raise KeyError(
                f"Unexpected Stage II delta keys: {incompatible.unexpected_keys}"
            )
        print(
            f"[Test] loaded compact Stage II delta: "
            f"{len(ckpt['gen_model_delta'])} tensors"
        )
    elif "gen_model" in ckpt:
        load_strip_module(generator, ckpt["gen_model"], strict=False)
        print("[Test] loaded legacy full Generator checkpoint.")
    else:
        raise KeyError("Checkpoint must contain `gen_model_delta` or legacy `gen_model`.")

    for model in models:
        model.eval()


def normalize_frame_names(frame_names):
    names = []
    for item in frame_names:
        if isinstance(item, (list, tuple)):
            item = item[0]
        if isinstance(item, str):
            names.append(item)
        else:
            names.append(str(item))
    return names


def output_frame_path(output_dir, dataset_name, video_name, frame_name):
    stem, _ = os.path.splitext(frame_name)
    return os.path.join(output_dir, dataset_name, video_name, f"{stem}.png")


def save_config(args):
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "test_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def maybe_profile(
    args,
    models,
    data_vis_in,
    base_input,
    detail_input,
    feature_f_b_5d,
    feature_f_d_5d,
    data_vis,
    decoder_input,
    fea_gen,
):
    if not args.profile:
        return None
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Install thop or remove --profile to run inference without FLOPs stats.") from exc

    encoder, decoder, base, detail, generator = models
    flops_1, params_1 = profile(encoder, inputs=(data_vis_in,), verbose=False)
    flops_2, params_2 = profile(base, inputs=(base_input,), verbose=False)
    flops_3, params_3 = profile(detail, inputs=(detail_input,), verbose=False)
    flops_4, params_4 = profile(generator, inputs=(feature_f_b_5d, feature_f_d_5d, data_vis), verbose=False)
    flops_5, params_5 = profile(decoder, inputs=(decoder_input, fea_gen, None, True), verbose=False)
    flops = float(flops_1 * 2 + flops_2 + flops_3 + flops_4 + flops_5)
    params = float(params_1 + params_2 + params_3 + params_4 + params_5)
    return {
        "flops": flops,
        "params": params,
        "gflops": flops / 1e9,
        "mparams": params / 1e6,
        "encoder_flops": float(flops_1 * 2),
        "base_fuse_flops": float(flops_2),
        "detail_fuse_flops": float(flops_3),
        "generator_flops": float(flops_4),
        "decoder_flops": float(flops_5),
        "encoder_params": float(params_1),
        "base_fuse_params": float(params_2),
        "detail_fuse_params": float(params_3),
        "generator_params": float(params_4),
        "decoder_params": float(params_5),
    }


def run_inference(args):
    global DataLoader, ToPILImage, torch, tqdm

    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToPILImage
    from tqdm import tqdm

    from utils.utils_ import set_random_seed

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)
    save_config(args)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    models = build_models(args, device)
    load_checkpoint(args.ckpt_path, models, device)
    encoder, decoder, base, detail, generator = models
    to_pil = ToPILImage()

    total_frames = 0
    total_infer_time = 0.0
    profile_summary = None
    dataset_name = args.dataset.lower()
    stream_state = None
    prev_vis_frame = None
    current_video = None
    print(f"[Test] dataset={args.dataset}, samples={len(dataset)}, device={device}")
    print(f"[Test] checkpoint={args.ckpt_path}")

    with torch.no_grad():
        for sample_idx, (video_name, frame_names, vis_seq, ir_seq) in enumerate(tqdm(loader, desc="Stage2 test")):
            if args.max_samples is not None and sample_idx >= args.max_samples:
                break
            if args.batch_size != 1:
                raise ValueError("Stage2 open test currently expects --batch_size 1.")

            video_name = video_name[0]
            if video_name != current_video:
                stream_state = None
                prev_vis_frame = None
                current_video = video_name
            frame_names = normalize_frame_names(frame_names)
            data_vis = vis_seq.to(device, non_blocking=True)
            data_ir = ir_seq.to(device, non_blocking=True)
            b, t, c, h, w = data_ir.shape

            start = time.time()
            data_vis_in = data_vis.view(b * t, c, h, w) + args.add_input_offset
            data_ir_in = data_ir.view(b * t, c, h, w) + args.add_input_offset

            feature_v_b, feature_v_d, _ = encoder(data_vis_in)
            feature_i_b, feature_i_d, _ = encoder(data_ir_in)
            base_input = feature_i_b + feature_v_b
            detail_input = feature_i_d + feature_v_d
            feature_f_b = base(base_input)
            feature_f_d = detail(detail_input)

            feature_f_b_5d = feature_f_b.view(b, t, -1, h, w)
            feature_f_d_5d = feature_f_d.view(b, t, -1, h, w)
            if args.no_stream_state:
                fea_gen, _ = generator(feature_f_b_5d, feature_f_d_5d, data_vis)
            else:
                fea_gen, _, stream_state = generator(
                    feature_f_b_5d,
                    feature_f_d_5d,
                    data_vis,
                    initial_feat_prop=stream_state,
                    prev_vis_frame=prev_vis_frame,
                    return_state=True,
                )
                stream_state = [state.detach() for state in stream_state]
                prev_vis_frame = data_vis[:, -1].detach()
            if not args.no_detail_residual:
                detail_pair = torch.cat((feature_v_d, feature_i_d), dim=1).view(b, t, -1, h, w)
                fea_gen = fea_gen + detail_pair
            fea_gen = fea_gen.view(b * t, -1, h, w)

            decoder_input = data_vis_in * 0.5 + data_ir_in * 0.5
            if profile_summary is None:
                profile_summary = maybe_profile(
                    args,
                    models,
                    data_vis_in,
                    base_input,
                    detail_input,
                    feature_f_b_5d,
                    feature_f_d_5d,
                    data_vis,
                    decoder_input,
                    fea_gen,
                )

            data_fuse, _ = decoder(
                decoder_input,
                fea_gen,
                detail_feature=None,
                is_2dfusion=True,
            )
            chunk_time = time.time() - start
            total_infer_time += chunk_time
            data_fuse_5d = data_fuse.view(b, t, -1, h, w)
            per_frame_time = chunk_time / max(len(frame_names), 1)

            for i, frame_name in enumerate(frame_names):
                fused_out_path = output_frame_path(args.output_dir, dataset_name, video_name, frame_name)
                if args.skip_existing and os.path.exists(fused_out_path):
                    continue
                os.makedirs(os.path.dirname(fused_out_path), exist_ok=True)
                fused = data_fuse_5d.squeeze(0)[i].cpu().clamp(0, 1)
                fused_p = to_pil(fused).convert("L")
                fused_p.save(fused_out_path)
                total_frames += 1

    avg_time = total_infer_time / max(total_frames, 1)
    log = {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "ckpt_path": args.ckpt_path,
        "output_dir": args.output_dir,
        "seq_len": args.seq_len,
        "overlap": args.overlap,
        "device": str(device),
        "total_frames_saved": total_frames,
        "total_infer_time_sec": total_infer_time,
        "avg_time_per_saved_frame_sec": avg_time,
        "profile": profile_summary,
    }
    log_path = os.path.join(args.output_dir, "test_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"[Done] saved frames: {total_frames}")
    print(f"[Done] avg time/frame: {avg_time:.6f}s")
    print(f"[Done] log: {log_path}")


if __name__ == "__main__":
    run_inference(parse_args())
