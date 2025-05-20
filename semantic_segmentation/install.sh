#!/bin/bash
set -e

echo "Installing PyTorch 2.0.0 + CUDA 11.8..."
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

echo "Installing OpenMIM..."
pip install openmim==0.3.9

echo "Installing mmengine and mmcv (2.0.0) via mim..."
mim install mmengine
mim install mmcv==2.0.0

echo "Installing imagecorruptions from GitHub..."
pip install git+https://github.com/bethgelab/imagecorruptions.git

echo "Installing local mmsegmentation package in editable mode..."
cd mmsegmentation
pip install -v -e .

echo "Installing local mmsegmentation ops package..."
cd ops
pip install -e .
cd ..

echo "Installing local mmsegmentation ops_dcnv3 package..."
cd ops_dcnv3
pip install -e .
cd ../..

echo "All done!"
