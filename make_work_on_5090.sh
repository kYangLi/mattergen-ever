# Source CUDA 12.9
#export PATH=/ssd/cuda12.9/install/bin:$PATH
#export LD_LIBRARY_PATH=/ssd/cuda12.9/install/lib64:$LD_LIBRARY_PATH

# Install current pkg
uv pip install -e .

# Install torch 2.8.0
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129 -U
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html -U
uv pip install torch_geometric -U
uv pip install "numpy<2.0" -U

