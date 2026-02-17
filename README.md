# VGGT Pytorch Inference (WIP)
Repository for running the VGGT model in PyTorch
![vggt](https://github.com/user-attachments/assets/ee3b24be-b97c-45fc-9361-ee686d447014)

## TODO
 - Add feature tracking (DeDoDe v2)
 - Add Bundle Adjustment from VGGSfM 
 - Add masking? (Sky, vehicles...) 

## Installation

> **Note:** This project depends on the official [facebookresearch/vggt](https://github.com/facebookresearch/vggt) package, which is automatically installed from GitHub during setup. Make sure you have **git** installed on your system before running the installation commands below.

### Option1: pip installation
```bash
git clone https://github.com/ibaiGorordo/vggt-pytorch-inference.git
cd vggt-pytorch-inference
pip install -r requirements.txt
```

### Option2: uv installation
Recommended but requires uv to be installed (https://docs.astral.sh/uv/getting-started/installation/)
```bash
git clone https://github.com/ibaiGorordo/vggt-pytorch-inference.git
cd vggt-pytorch-inference
uv sync
```

- Additionally, to activate the uv environment: `source .venv/bin/activate` in macOS and Linux or `.venv\Scripts\activate` in Windows

### Troubleshooting

If you encounter `ModuleNotFoundError: No module named 'vggt.models'`, the `vggt` dependency was not installed correctly. Try installing it manually:

```bash
pip install git+https://github.com/facebookresearch/vggt.git
```

## Examples:

```bash
python example.py
```

## License
The code is taken from the official [VGGT repository](https://github.com/facebookresearch/vggt) which is distributed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
See [LICENSE](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) for more information.


### References:
- vggt: https://github.com/facebookresearch/vggt
- VGGSfM: https://github.com/facebookresearch/vggsfm
