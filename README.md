# MambaLF: An Efficient Local Feature Extraction and Matching with State Space Model

## Getting started

### 1. Installation

MambaLF is developed based on `torch==2.2.0` `python==12.2` and `CUDA Version==12.2`.

#### 2.Clone Project 

```bash
git clone https://github.com/TakeoffC/MambaLF.git
```

#### 3.Create and activate a conda environment.
```bash
conda create -n mambalf -y python=3.10
conda activate mambalf
```

#### 4.Install Dependencies

```bash
pip install -r requirements.txt
```
Please also refer to "https://github.com/MzeroMiko/VMamba" and install the relevant dependencies for VMamba.

#### 5. Prepare MSCOCO2017 Dataset
Download the COCO dataset:
```
cd datasets/COCO/
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```
Prepare the training file:
```
python datasets/prepare_coco.py --raw_dir datasets/COCO/train2017/ --saved_dir datasets/COCO/ 
```

#### 6. Training MambaLF
To train the model on COCO dataset, run:
```
python main.py --train_root datasets/COCO/train2017/ --train_txt datasets/COCO/train2017.txt
```

#### 7. Evaluating MambaLF
For the evaluation of **MambaLF**, we use the **"image-matching-toolbox"** to compare its performance with other methods on the **HPatches** and **RDNIM** datasets.  

The evaluation toolbox and datasets can be found at the following links:  
- **image-matching-toolbox**: [https://github.com/GrumpyZhou/image-matching-toolbox](https://github.com/GrumpyZhou/image-matching-toolbox)  
- **HPatches dataset**: [https://github.com/hpatches](https://github.com/hpatches)  
- **RDNIM dataset**: [https://github.com/rpautrat/LISRD](https://github.com/rpautrat/LISRD)

## Acknowledgement
This repo is modified from open source codebase [KP2D](https://github.com/TRI-ML/KP2D) and [LAnet](https://github.com/wangch-g/lanet) .The selective-scan from [VMamba](https://github.com/MzeroMiko/VMamba).

