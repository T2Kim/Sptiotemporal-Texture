Spatiotemporal Texture Reconstruction
========================

### This repository contains the official implementation of the following paper:

> **Spatiotemporal Texture Reconstruction for Dynamic Objects Usinga Single RGB-D Camera (Eurographics 2021)**
> ![Teaser](./teaser.pdf)
### Requirements
*Windows 10
*CUDA >= 10.0
*python >= 3.6
*MSVC++ >= 14.2

### Quick Start
0. Parameterization
UVAtlas code: https://github.com/microsoft/UVAtlas
1. Foreground extraction
You can find foreground extraction code in https://github.com/csaishih/foreground-extraction
2. Global texture coordinate optimization
Build and execute TextureMappingNonRigid project.
3. Prepare labeling
In ./Similarity folder,
python ./simchek.py
python ./variance.py
In SHOT folder,
python ./shot.py
4. Labeling
Re-execute TextureMappingNonRigid project.
5. Color correction
We use Texture Stitching program: https://github.com/mkazhdan/TextureSignalProcessing
6. View result
In conf.json file, convert "is_viewer" false to true, and execute.