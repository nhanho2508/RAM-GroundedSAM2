
### **Setting Up RAM - Recognize Anything** ###
0. Link Checkpoint: https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth
1. Create and activate a Conda environment:

```bash
conda create -n recognize-anything python=3.8 -y
conda activate recognize-anything
```


2. Or, for development, you may build from source:

```bash
git clone https://github.com/xinyu1205/recognize-anything.git
cd recognize-anything
pip install -e .
```

### **Setting Up RAM - Grounded - SAM -2** ###
#### Installation
0. Di chuyển đến folder Grounded_SAM_2
1. Download the pretrained `SAM 2` checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
```

2. Download the pretrained `Grounding DINO` checkpoints:

```bash
cd gdino_checkpoints
bash download_ckpts.sh
```
3. 
Install PyTorch environment first. We use `python=3.10`, as well as `torch >= 2.3.1`, `torchvision>=0.18.1` and `cuda-12.1` in our environment to run this demo. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. You can easily install the latest version of PyTorch as follows:

```bash
pip3 install torch torchvision torchaudio
```
Since we need the CUDA compilation environment to compile the `Deformable Attention` operator used in Grounding DINO, we need to check whether the CUDA environment variables have been set correctly (which you can refer to [Grounding DINO Installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for more details). You can set the environment variable manually as follows if you want to build a local GPU environment for Grounding DINO to run Grounded SAM 2:

```bash
export CUDA_HOME=/path/to/cuda-12.1/ 
```
(Nếu đã có CUDA_HOME trong eviroment variable thì không cần)


Install `Segment Anything 2`:

```bash
pip install -e .
```


Install `Grounding DINO`:

```bash
pip install --no-build-isolation -e grounding_dino
```
(Nếu không chạy lệnh trên được thử chạy lệnh sau)
```bash
pip install --no-build-isolation -e grounding_dino --config-settings editable_mode=compat
```

### Inference Segmentation and Edge Detection
Trong file infer_segment_edge.py, dùng absolute path tương ứng với các checkpoint tương ứng
```bash
SAM2_CHECKPOINT = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "C:/Users/ADMIN/source/repos/RAM+GroundedSAM2/Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth"
```
```bash
model_ram = ram_plus(pretrained='C:/Users/ADMIN/Downloads/ram_plus_swin_large_14m.pth', image_size=IMAGE_SIZE, vit='swin_l')
```

With Places2, ImageNet Dataset
```bash
python infer_segment_edge.py --source-folder C:\Users\ADMIN\Downloads\Places2_1\test_256 --target-folder outputs/grounded_sam2_dir_demo --kernel-size 7 --min-threshold 20 --max-threshold 50
```
With FFHQ Dataset
```bash
python infer_segment_edge.py --source-folder C:\Users\ADMIN\Downloads\FFHQ --target-folder outputs/grounded_sam2_dir_demo --kernel-size 7 --min-threshold 5 --max-threshold 10
```

--kernel-size: Size of Median kernel
--max-threshold , --min-threshold: Threshold of Canny Edge Detection

FFHQ: kernel size (7), min threshold (5), max threshold (10)

Places, ImageNet: kernel size (7), min threshold (20), max threshold (50)