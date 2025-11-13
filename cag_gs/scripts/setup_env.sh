conda create -yn cag-gs python=3.10
conda activate cag-gs

pip install -r requirements/pytorch.txt
pip install -r requirements/lightning.txt
pip install -r requirements/common.txt

pip install -r requirements/fused_ssim.txt
pip install -r requirements/tcnn.txt
pip install -r requirements/torch_scatter.txt
pip install -r requirements/pytorch3d.txt
pip install -r requirements/gsplat.txt

pip install -r requirements/sam2.txt