import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instance_corners import (
    compose_full_mask,
    extract_instances,
    process_instance,
)
from models.segmentation_model import SegmentationModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal UniBuild DINOv3-Base HR-DPT GeoTIFF inference with optional instance polygonization."
    )
    parser.add_argument("--input", required=True, help="Input RGB GeoTIFF. Put example inputs under data/.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/unibuild_dinov3_base_hrdpt.pth",
        help="Trained UniBuild DINOv3-Base HR-DPT checkpoint.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="Optional standalone DINOv3-Base pretrained checkpoint. Not needed when --checkpoint is a full UniBuild checkpoint.",
    )
    parser.add_argument("--output-dir", default="data/outputs", help="Output directory.")
    parser.add_argument("--name", default=None, help="Output stem. Defaults to input file stem.")
    parser.add_argument("--dinov3-variant", default="base", choices=["base"])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=[1, 2, 3])
    parser.add_argument("--scale-mode", default="auto", choices=["auto", "uint8", "minmax"])
    parser.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--save-prob", action="store_true", help="Also save probability GeoTIFF.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--upsample-to-gsd",
        type=float,
        default=None,
        help="If set and input pixel size is coarser than this value, infer on a bilinearly upsampled VRT grid.",
    )

    parser.add_argument(
        "--polygonize",
        "--regularize",
        dest="polygonize",
        action="store_true",
        help="Extract direction-aware building-instance polygons and corners.",
    )
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--min-instance-area",
        type=int,
        default=9,
        help="Ignore connected building instances smaller than this many pixels.",
    )
    return parser.parse_args()


def build_model_cfg(args):
    return {
        "backbone": {"name": "dinov3", "variant": args.dinov3_variant, "pretrained": args.backbone_checkpoint},
        "decoder": {"name": "hlrdpt"},
    }


def load_model(args, device):
    model = SegmentationModel(build_model_cfg(args), nclass=2).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
    model.eval()
    return model


def compute_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_weight(tile_size):
    w = np.hanning(tile_size).astype(np.float32)
    return np.clip(np.outer(w, w).astype(np.float32), 1e-6, None)


def scale_rgb(rgb, mode):
    rgb = rgb.astype(np.float32)
    if mode == "uint8" or (mode == "auto" and np.nanmax(rgb) <= 255.0):
        return np.clip(rgb / 255.0, 0.0, 1.0)

    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[i]
        valid = np.isfinite(band)
        if not np.any(valid):
            continue
        lo = float(np.min(band[valid]))
        hi = float(np.max(band[valid]))
        if hi > lo:
            out[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def read_patch(src, row, col, tile_size, bands, scale_mode, mean, std):
    h = min(tile_size, src.height - row)
    w = min(tile_size, src.width - col)
    window = rasterio.windows.Window(col, row, w, h)
    rgb = src.read(bands, window=window, boundless=False)
    rgb = scale_rgb(rgb, scale_mode)
    if h != tile_size or w != tile_size:
        padded = np.zeros((3, tile_size, tile_size), dtype=np.float32)
        padded[:, :h, :w] = rgb
        rgb = padded
    patch = torch.from_numpy(rgb)
    mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(std, dtype=torch.float32)[:, None, None]
    return (patch - mean) / std, h, w


@torch.no_grad()
def predict_batch(model, batch):
    outputs = model(batch)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    if logits.shape[-2:] != batch.shape[-2:]:
        logits = F.interpolate(logits, size=batch.shape[-2:], mode="bilinear", align_corners=False)
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0]).float()
    return torch.softmax(logits, dim=1)[:, 1].float()


def make_output_profile(src, dtype, count=1):
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        width=src.width,
        height=src.height,
        transform=src.transform,
        count=count,
        dtype=dtype,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=None,
        BIGTIFF="IF_SAFER",
    )
    return profile


def output_paths(args):
    out_dir = Path(args.output_dir)
    stem = args.name or Path(args.input).stem
    return {
        "prob": out_dir / f"{stem}_prob.tif",
        "mask": out_dir / f"{stem}_mask.tif",
        "polygonized_mask": out_dir / f"{stem}_mask_polygonized.tif",
        "vector": out_dir / f"{stem}_buildings.gpkg",
    }


def maybe_upsampled_source(src, target_gsd):
    if target_gsd is None:
        return nullcontext(src)
    xres = abs(src.transform.a)
    yres = abs(src.transform.e)
    if max(xres, yres) <= target_gsd:
        return nullcontext(src)
    scale_x = xres / target_gsd
    scale_y = yres / target_gsd
    width = int(round(src.width * scale_x))
    height = int(round(src.height * scale_y))
    transform = src.transform * src.transform.scale(src.width / width, src.height / height)
    print(f"Upsampling VRT from GSD ({xres:.3f}, {yres:.3f}) to about {target_gsd:.3f}.")
    return (
        WarpedVRT(
            src,
            crs=src.crs,
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
        )
    )


def run_inference(args):
    paths = output_paths(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        outputs = [paths["mask"]]
        if args.save_prob:
            outputs.append(paths["prob"])
        if any(path.exists() for path in outputs):
            raise FileExistsError(f"Output exists. Use --overwrite to regenerate: {outputs}")

    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    with rasterio.open(args.input) as base_src:
        with maybe_upsampled_source(base_src, args.upsample_to_gsd) as src:
            if max(args.rgb_bands) > src.count or min(args.rgb_bands) < 1:
                raise ValueError(f"--rgb-bands {args.rgb_bands} is invalid for input band count={src.count}")

            y_starts = compute_starts(src.height, args.tile_size, args.stride)
            x_starts = compute_starts(src.width, args.tile_size, args.stride)
            weight = blend_weight(args.tile_size)
            prob_sum = np.zeros((src.height, src.width), dtype=np.float32)
            weight_sum = np.zeros_like(prob_sum)

            patches, records = [], []
            total = len(y_starts) * len(x_starts)
            done = 0

            def flush():
                nonlocal done, patches, records
                if not patches:
                    return
                batch = torch.stack(patches, dim=0).to(device)
                probs = predict_batch(model, batch).cpu().numpy()
                for prob, (row, col, h, w) in zip(probs, records):
                    prob_sum[row:row + h, col:col + w] += prob[:h, :w] * weight[:h, :w]
                    weight_sum[row:row + h, col:col + w] += weight[:h, :w]
                done += len(records)
                print(f"Processed {done}/{total} windows", flush=True)
                patches, records = [], []

            for row in y_starts:
                for col in x_starts:
                    patch, h, w = read_patch(
                        src,
                        row=row,
                        col=col,
                        tile_size=args.tile_size,
                        bands=args.rgb_bands,
                        scale_mode=args.scale_mode,
                        mean=args.mean,
                        std=args.std,
                    )
                    patches.append(patch)
                    records.append((row, col, h, w))
                    if len(patches) >= args.batch_size:
                        flush()
            flush()

            prob = prob_sum / np.clip(weight_sum, 1e-6, None)
            mask = (prob >= args.threshold).astype(np.uint8)

            if args.save_prob:
                with rasterio.open(paths["prob"], "w", **make_output_profile(src, "float32")) as dst:
                    dst.write(prob.astype(np.float32), 1)
                print(f"Saved probability: {paths['prob']}")
            with rasterio.open(paths["mask"], "w", **make_output_profile(src, "uint8")) as dst:
                dst.write(mask, 1)
            print(f"Saved mask: {paths['mask']}")

    if args.polygonize:
        polygonize_mask(args, paths)


def build_polygonizer_args(args):
    return SimpleNamespace(
        connectivity=args.connectivity,
        min_instance_area=args.min_instance_area,
        bbox_padding=1,
        chain_approx="none",
        epsilon_ratio=0.0015,
        min_epsilon_px=1.5,
        max_epsilon_px=4.5,
        angle_threshold_deg=12.0,
        preserve_curves=True,
        curve_turn_threshold_deg=30.0,
        curve_window=3,
        max_collinear_distance_px=1.0,
        direction_threshold_deg=12.0,
        snap_to_dominant=True,
        max_snap_shift_px=3.0,
        direction_prune_iters=3,
        jagged_collapse=True,
        jagged_min_contour_vertices=10,
        jagged_min_run_vertices=5,
        jagged_max_run_vertices=12,
        jagged_short_edge_px=4.0,
        jagged_min_short_edge_ratio=0.70,
        jagged_max_mean_line_distance_px=1.5,
        jagged_min_direction_switches=2,
        jagged_collapse_iters=20,
    )


def contour_to_world_ring(contour, transform):
    ring = [
        tuple(
            float(value)
            for value in transform * (float(x) + 0.5, float(y) + 0.5)
        )
        for x, y in contour.reshape(-1, 2)
    ]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def extract_polygonized_instances(mask, args):
    source_mask = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    polygonizer_args = build_polygonizer_args(args)
    instances = []
    for instance_id, bbox_xywh, area, crop_mask in extract_instances(
        source_mask,
        polygonizer_args,
    ):
        instance = process_instance(
            instance_id,
            bbox_xywh,
            area,
            crop_mask,
            polygonizer_args,
        )
        if instance is not None:
            instances.append(instance)
    return instances


def write_instance_gpkg(path, instances, transform, crs):
    try:
        import fiona
    except ImportError as exc:
        raise ImportError(
            "Writing GeoPackage polygons requires Fiona: pip install fiona"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    schema = {
        "geometry": "Polygon",
        "properties": {
            "inst_id": "int",
            "area_px": "int",
            "vertices": "int",
            "angle_deg": "float",
        },
    }
    open_kwargs = {
        "driver": "GPKG",
        "schema": schema,
        "layer": "buildings",
    }
    if crs is not None:
        open_kwargs["crs_wkt"] = crs.to_wkt()

    feature_count = 0
    with fiona.open(path, "w", **open_kwargs) as sink:
        for instance in instances:
            records = instance.contour_records
            for contour in records:
                if contour.is_hole:
                    continue
                shell = contour_to_world_ring(contour.contour_global, transform)
                if len(shell) < 4:
                    continue
                holes = [
                    contour_to_world_ring(candidate.contour_global, transform)
                    for candidate in records
                    if candidate.is_hole
                    and candidate.parent_index == contour.contour_index
                    and candidate.depth == contour.depth + 1
                ]
                holes = [ring for ring in holes if len(ring) >= 4]
                sink.write(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [shell, *holes],
                        },
                        "properties": {
                            "inst_id": int(instance.instance_id),
                            "area_px": int(instance.area),
                            "vertices": int(contour.vertex_count),
                            "angle_deg": float(contour.dominant_angle_deg),
                        },
                    }
                )
                feature_count += 1
    return feature_count


def polygonize_mask(args, paths):
    with rasterio.open(paths["mask"]) as source:
        mask = source.read(1)
        transform = source.transform
        crs = source.crs
        profile = source.profile.copy()

    instances = extract_polygonized_instances(mask, args)
    if not instances:
        raise ValueError(f"No building instances found in {paths['mask']}")

    feature_count = write_instance_gpkg(
        paths["vector"],
        instances,
        transform,
        crs,
    )
    print(
        f"Saved {feature_count} direction-aware building polygons: "
        f"{paths['vector']}"
    )

    polygonized_mask = (compose_full_mask(mask.shape, instances) > 0).astype(np.uint8)
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(paths["polygonized_mask"], "w", **profile) as destination:
        destination.write(polygonized_mask, 1)
    print(f"Saved polygonized mask: {paths['polygonized_mask']}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
