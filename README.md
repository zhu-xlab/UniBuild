# UniBuild Standalone Inference

This folder contains the minimum code needed to run the trained **UniBuild DINOv3-Base HR-DPT** checkpoint for building mask inference on RGB optical remote sensing GeoTIFFs. It only keeps the DINOv3-Base backbone and HR-DPT decoder needed for the released model. It can also extract direction-aware building-instance polygons and corners from the predicted mask.

![UniBuild building extraction example](figures/ood_google_crop.png)

The folder is self-contained for inference and polygonization: it does not import code from the full UniBuild repository and does not require Building-Regulariser.

## Folder Layout

```text
unibuild_inference/
  infer_geotiff.py          # sliding-window GeoTIFF inference + optional polygonization
  instance_corners/         # self-contained corner extraction and polygonization
    extractor.py
    polygonizer.py
  requirements.txt
  README.md
  checkpoints/              # put model checkpoints here
  data/                     # put input RGB GeoTIFFs and outputs here
    outputs/                # default output directory
  models/                   # minimal local model code copied from UniBuild
```

## Checkpoints

Download the trained UniBuild checkpoint from either mirror:

- [Google Drive](https://drive.google.com/drive/folders/1YZ-bbXoZ1rOKQayRL79bLdhIE_CcU3K9?usp=drive_link)
- [Baidu Netdisk](https://pan.baidu.com/s/1ViHEDvAkf_VjdP8gAixANw) (extraction code: `unbd`)

Place the downloaded checkpoint under `checkpoints/` with the following filename:

```text
checkpoints/
  unibuild_dinov3_base_hrdpt.pth
```

This full checkpoint already contains the DINOv3-Base backbone and HLRDPT decoder weights, so a separate DINOv3 pretrained backbone checkpoint is not required for normal inference.
If you intentionally want to initialize the backbone before loading a partial checkpoint, pass it with `--backbone-checkpoint`.

## Installation

Create or activate a Python environment with PyTorch installed, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you only need raster mask inference and do not need vector footprint generation, OpenCV and Fiona can be omitted:

```bash
pip install numpy torch rasterio
```

## Inference

Run sliding-window inference on an RGB GeoTIFF:

```bash
python infer_geotiff.py \
  --input data/input_rgb.tif \
  --polygonize
```

`--polygonize` is optional. Keep it to extract direction-aware building-instance polygons and corners; remove it to generate only the binary building mask.

The binary mask is always saved:

```text
data/outputs/input_rgb_mask.tif
```

With `--polygonize`, the command additionally saves:

```text
data/outputs/input_rgb_buildings.gpkg
data/outputs/input_rgb_mask_polygonized.tif
```

The polygonization strategy separates connected instances, simplifies contours under dominant-direction constraints, and merges short corner transitions only when both turns share the same direction. The short-edge merge threshold is `8.0` pixels. Use `--min-instance-area` and `--connectivity` to control instance filtering and connectivity.

## Notes

- Input imagery should be RGB optical remote sensing imagery.
- The default RGB bands are `1 2 3`. Use `--rgb-bands` if your GeoTIFF stores RGB in a different order.
- The output GeoTIFFs preserve the inference grid CRS, transform, bounds, and size. If `--upsample-to-gsd` is used, the outputs are written on the upsampled grid.
- Use `--overwrite` to regenerate existing outputs.
