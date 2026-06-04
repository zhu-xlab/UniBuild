# UniBuild Standalone Inference

This folder contains the minimum code needed to run the trained **UniBuild DINOv3-Base HR-DPT** checkpoint for building mask inference on RGB optical remote sensing GeoTIFFs. It only keeps the DINOv3-Base backbone and HLRDPT decoder needed for the released model. It can also optionally polygonize and regularize the predicted mask into vectorized building footprints.

The folder is self-contained for inference: run commands from inside `unibuild_inference/`.

## Folder Layout

```text
unibuild_inference/
  infer_geotiff.py          # sliding-window GeoTIFF inference + optional regularization
  requirements.txt
  README.md
  checkpoints/              # put model checkpoints here
  outputs/                  # default output directory
  models/                   # minimal local model code copied from UniBuild
```

## Checkpoints

Place the trained UniBuild checkpoint under `unibuild_inference/checkpoints/`:

```text
checkpoints/
  best_dinov3_base_hlrdpt-512-Multi-12-ce-saddle-stdir_lamSAD-15_lamST-0p5.pth
```

This full checkpoint already contains the DINOv3-Base backbone and HLRDPT decoder weights, so a separate DINOv3 pretrained backbone checkpoint is not required for normal inference.
If you intentionally want to initialize the backbone before loading a partial checkpoint, pass it with `--backbone-checkpoint`.

## Installation

Create or activate a Python environment with PyTorch installed, then install the remaining dependencies:

```bash
cd unibuild_inference
pip install -r requirements.txt
```

If you only need raster mask inference and do not need vector footprint generation, the regularization dependencies can be omitted:

```bash
pip install numpy torch rasterio shapely
```

## Building Mask Inference

Run sliding-window inference on an RGB GeoTIFF:

```bash
cd unibuild_inference

python infer_geotiff.py \
  --input /path/to/input_rgb.tif \
  --checkpoint checkpoints/best_dinov3_base_hlrdpt-512-Multi-12-ce-saddle-stdir_lamSAD-15_lamST-0p5.pth \
  --output-dir outputs \
  --save-prob
```

Outputs:

```text
outputs/input_rgb_mask.tif    # binary building mask, 0 background / 1 building
outputs/input_rgb_prob.tif    # optional float32 building probability
```

For images coarser than 1 m GSD, upsample to 1 m during inference:

```bash
python infer_geotiff.py \
  --input /path/to/coarse_rgb.tif \
  --checkpoint checkpoints/best_dinov3_base_hlrdpt-512-Multi-12-ce-saddle-stdir_lamSAD-15_lamST-0p5.pth \
  --output-dir outputs \
  --upsample-to-gsd 1.0
```

## Mask + Building Regularization

To generate both raster masks and vectorized regularized building footprints:

```bash
python infer_geotiff.py \
  --input /path/to/input_rgb.tif \
  --checkpoint checkpoints/best_dinov3_base_hlrdpt-512-Multi-12-ce-saddle-stdir_lamSAD-15_lamST-0p5.pth \
  --output-dir outputs \
  --regularize \
  --simplify-tolerance 2.0 \
  --parallel-threshold 2.0 \
  --min-area 8
```

Outputs:

```text
outputs/input_rgb_mask.tif                  # raw binary mask
outputs/input_rgb_buildings_regularized.gpkg # regularized vector footprints
outputs/input_rgb_mask_regularized.tif       # rasterized regularized footprints
```

Useful regularization options:

```bash
--allow-circles
--circle-threshold 0.85
--allow-45-degree
--simplify-tolerance 2.0
--parallel-threshold 2.0
--min-area 8
```

For meter-based CRS, `--simplify-tolerance` around 2-3 times the pixel size usually reduces stair-step artifacts more strongly.

## Notes

- Input imagery should be RGB optical remote sensing imagery.
- The default RGB bands are `1 2 3`. Use `--rgb-bands` if your GeoTIFF stores RGB in a different order.
- The output GeoTIFFs preserve the inference grid CRS, transform, bounds, and size. If `--upsample-to-gsd` is used, the outputs are written on the upsampled grid.
- Use `--overwrite` to regenerate existing outputs.
