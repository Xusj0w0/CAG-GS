# [AAAI26] CAG-GS: Consistent Anchor Guided Gaussian Splatting for Large-Scale Scene Rendering

This repo contains the official implementation of [**CAG-GS**](https://ojs.aaai.org/index.php/AAAI/article/view/38120), a Gaussian Splatting method for large-scale scene rendering.

## 📌 TODO

- [x] Release the codes for training and evaluation.
- [ ] Release the checkpoints.

## 🚀 Quick Start

### Installation

Please create virtual environment as follows:
```bash
conda create -n cag-gs python=3.10.12 -y
conda activate cag-gs
bash cag_gs/scripts/setup_env.sh
```

### Data Preparation

1. Download and process datasets following [CityGS](https://github.com/Linketic/CityGaussian). The propocessed dataset folder structure is:
    ```
    ├─ citygs
      ├─ building
          ├─ images
          ├─ sparse
            ├─ 0
              ├─ cameras.bin
              ├─ images.bin
              ├─ points3D.bin
      ├─ rubble
      ├─ residence
      ├─ sci-art
      ├─ mc_aerial
   ```

2. Downsample images and modify sparse model accordingly. Take the Building scene as an example:
    ```bash
    python cag_gs/tools/convert_dataset.py \
      --input dataset/citygs/building \
      --output_dir dataset/benchmark/building \
      --down_sample_factor 4
    ```
    **[Note]** For MatrixCity-Aerial, please replace `--down_sample_factor=4` with `--rescale_width=1600`. Additionally, include `--prefix=val_` to avoid filename conflicts, as both the training and validation sets share the same indexing.

3. Compute SAM2 embeddings. Download checkpoint from [sam2.1_hiera_large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) and compute the SAM2 image embeddings for the datasets. Take the Building scene as an example:
    ```bash
    python cag_gs/tools/get_sam2_embeddings.py dataset/benchmark/building
    ```
    The dataset folder structure for the Building scene is expected to be:
    ```
    ├─ building
      ├─ images
      ├─ sparse
      ├─ extra
        ├─ sam2_large_d32
    ```

## 📖 Citation
```
@article{Xu_Dong_2026,
  title={CAG-GS: Consistent Anchor Guided Gaussian Splatting for Large-scale Scene Rendering}, 
  volume={40}, 
  number={14}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Xu, Shijie and Dong, Qiulei}, 
  year={2026}, 
  pages={11388-11396} }
```

## 🎯 Acknowledgements

This repo is built upon [GaussianSplatting Lightning](https://github.com/yzslab/gaussian-splatting-lightning), [CityGS](https://github.com/Linketic/CityGaussian), and [Octree-GS](https://github.com/city-super/Octree-GS). Thanks for their great work!