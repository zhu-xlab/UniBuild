# UniBuild: Unified Building Mapping From Multi-Source Optical Remote Sensing Imagery

Official inference implementation of **UniBuild**, a unified building-extraction framework for multi-source RGB optical remote sensing imagery. UniBuild combines a DINOv3 backbone, a detail-preserving HR-DPT decoder, direction-aware boundary regularization, and saddle-aware suppression for separating adjacent buildings. The released model supports imagery from diverse sensors and resolutions up to 10 m GSD and produces building masks with optional GIS-compatible vector footprints.

## Paper

**UniBuild: Unified Building Mapping From Multi-Source Optical Remote Sensing Imagery With Detail Decoding and Geometry Regularization**

Wei Huang, Chenying Liu, Yilei Shi, and Xiao Xiang Zhu

[arXiv:2609.37031](https://arxiv.org/abs/2609.37031)

![UniBuild building extraction example](figures/ood_google_crop.png)

## Highlights

- One model for multi-source RGB optical imagery spanning 0.05-10 m GSD.
- HR-DPT decoding for sharper boundaries, corners, and small buildings.
- Direction-aware and saddle-aware training objectives for regular boundaries and improved separation of adjacent buildings.
- Sliding-window GeoTIFF inference with georeferenced raster outputs and optional building polygonization.

## Model Checkpoint

Download the trained UniBuild checkpoint from either mirror:

- [Google Drive](https://drive.google.com/drive/folders/1YZ-bbXoZ1rOKQayRL79bLdhIE_CcU3K9?usp=drive_link)
- [Baidu Netdisk](https://pan.baidu.com/s/1ViHEDvAkf_VjdP8gAixANw) (extraction code: `unbd`)

Save it as:

```text
checkpoints/unibuild_dinov3_base_hrdpt.pth
```

The full checkpoint contains both the DINOv3-Base backbone and HR-DPT decoder weights. A separate backbone checkpoint is only needed when loading a partial UniBuild checkpoint; provide it with `--backbone-checkpoint`.

## Installation

Create or activate a Python environment with PyTorch installed, then run:

```bash
pip install -r requirements.txt
```

## Inference

Place an RGB GeoTIFF under `data/` and run:

```bash
python infer_geotiff.py \
  --input data/input_rgb.tif \
  --device cuda \
  --polygonize
```

Use `--device cpu` for CPU inference. `--polygonize` is optional; omit it to generate only the binary building mask. For imagery coarser than 1 m GSD, add `--upsample-to-gsd 1` to run inference on a 1 m grid.

### Outputs

The binary mask is always saved to:

```text
data/outputs/input_rgb_mask.tif
```

With `--polygonize`, two additional files are generated:

```text
data/outputs/input_rgb_buildings.gpkg
data/outputs/input_rgb_mask_polygonized.tif
```

The polygonization pipeline separates connected instances, simplifies contours under dominant-direction constraints, and merges short corner transitions when both turns share the same direction. Use `--min-instance-area` and `--connectivity` to control instance filtering and connectivity.

## Training Datasets

The released model was jointly trained on 12 RGB optical building-extraction datasets spanning 0.05-10 m GSD:

| Resolution group | Datasets | Sensor/source | GSD |
| --- | --- | --- | --- |
| High resolution | Potsdam | Aerial orthophoto | 0.05 m |
| High resolution | INRIA, LoveDA, SpaceNet2, LandCover.ai | Aerial imagery, Google Earth, WorldView-3, and orthophotos | 0.30 m |
| High resolution | OpenEarthMap (OEM) | Multi-source aerial, satellite, and UAV imagery | 0.25-0.50 m |
| High resolution | ORBITaL-Net | Maxar VHR, mainly WorldView-2/3 | 0.47 m |
| High resolution | Alabama, WHU-Mix | Bing Maps and LINZ aerial imagery | 0.50 m |
| High resolution | GF-7 | GaoFen-7 satellite imagery | 0.65 m |
| Low resolution | Planet | PlanetScope imagery | 4.80 m |
| Low resolution | Sentinel-2 (ST-2) | Sentinel-2 RGB imagery | 10 m |

The Planet and Sentinel-2 datasets were constructed over globally distributed urban areas using OpenStreetMap-derived building annotations. See the paper for dataset splits and full experimental details.

## Repository Layout

```text
.
├── infer_geotiff.py          # Sliding-window GeoTIFF inference
├── instance_corners/         # Corner extraction and polygonization
├── models/                   # DINOv3-Base and HR-DPT model code
├── checkpoints/              # Model checkpoints
├── data/                     # Input imagery and generated outputs
├── figures/                  # README assets
└── requirements.txt
```

## Notes

- Input imagery must contain RGB optical bands. The default band order is `1 2 3`; use `--rgb-bands` for a different order.
- RGB values are normalized directly from 0-255 to 0-1; values outside the input range are clipped.
- Output GeoTIFFs preserve the inference grid's CRS, transform, bounds, and size.
- Use `--save-prob` to save the probability map and `--overwrite` to replace existing outputs.

## Citation

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
