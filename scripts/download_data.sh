#!/bin/bash
pip install kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d qingyi/wm811k-wafer-map
unzip wm811k-wafer-map.zip -d ../data/
