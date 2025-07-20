conda create --prefix .venv python=3.12
conda activate .venv
conda install cuda=12.1 -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r requirements.txt

pip install 'tensorflow[and-cuda]'
pip uninstall torch torchvision torchaudio && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
