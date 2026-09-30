# UniBuild: Unified Building Mapping From Multi-Source Optical Remote Sensing Imagery

Official inference implementation of **UniBuild**, a unified building-extraction framework for multi-source RGB optical remote sensing imagery. UniBuild combines a DINOv3 backbone with a detail-preserving HR-DPT decoder, direction-aware boundary regularization, and saddle-aware suppression of false connections between adjacent buildings. A single model handles imagery from diverse sensors and resolutions up to 10 m GSD, producing building masks and optional GIS-compatible vector footprints.

## Paper

**UniBuild: Unified Building Mapping From Multi-Source Optical Remote Sensing Imagery With Detail Decoding and Geometry Regularization**

Wei Huang, Chenying Liu, Yilei Shi, and Xiao Xiang Zhu

[arXiv:2609.37031](https://arxiv.org/abs/2609.37031)

```bibtex
@misc{huang2026unibuild,
  title         = {UniBuild: Unified Building Mapping From Multi-Source Optical Remote Sensing Imagery With Detail Decoding and Geometry Regularization},
  author        = {Huang, Wei and Liu, Chenying and Shi, Yilei and Zhu, Xiao Xiang},
  year          = {2026},
  eprint        = {2609.37031},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.37031}
}
```

![UniBuild building extraction example](figures/ood_google_crop.png)

This repository contains the self-contained inference and polygonization code for the released **UniBuild DINOv3-Base HR-DPT** checkpoint. It does not require Building-Regulariser.

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

This full checkpoint already contains the DINOv3-Base backbone and HR-DPT decoder weights, so a separate DINOv3 pretrained backbone checkpoint is not required for normal inference.
If you intentionally want to initialize the backbone before loading a partial checkpoint, pass it with `--backbone-checkpoint`.

## Training Datasets

The released UniBuild model was jointly trained on 12 RGB optical building-extraction datasets spanning 0.05–10 m ground sampling distance (GSD):

| Resolution group | Datasets | Sensor/source | GSD |
| --- | --- | --- | --- |
| High resolution | Potsdam | Aerial orthophoto | 0.05 m |
| High resolution | INRIA, LoveDA, SpaceNet2, LandCover.ai | Aerial imagery, Google Earth, WorldView-3, and orthophotos | 0.30 m |
| High resolution | OpenEarthMap (OEM) | Multi-source aerial, satellite, and UAV imagery | 0.25–0.50 m |
| High resolution | ORBITaL-Net | Maxar VHR, mainly WorldView-2/3 | 0.47 m |
| High resolution | Alabama, WHU-Mix | Bing Maps and LINZ aerial imagery | 0.50 m |
| High resolution | GF-7 | GaoFen-7 satellite imagery | 0.65 m |
| Low resolution | Planet | PlanetScope imagery | 4.80 m |
| Low resolution | Sentinel-2 (ST-2) | Sentinel-2 RGB imagery | 10 m |

The Planet and Sentinel-2 datasets were constructed over globally distributed urban areas using OpenStreetMap-derived building annotations. See the paper for dataset splits and full experimental details.

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
