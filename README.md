# SfM Guided Dense Reconstruction  
## 1. Prepare depth estimation model  
```bash
cd Thirdparty/Depth-Anything-V2
pip install -r requirements.txt
mkdir checkpoints && cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

```
## 2. SfM Reconstruction  
```depth2ddd/utils/sfm.py```  
## 3. Dense Reconstruction  
```pipeline.py```  