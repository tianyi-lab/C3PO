# OLMoE Routing Weight Optimizer

This repository demonstrates how C3PO optimizing expert pathways to improve the model performance.

## Setup and Installation

### 1. Create Conda Environment

Create a new conda environment named C3PO and install the required packages:

```bash
# Create conda environment
conda create -n C3PO python=3.10 -y
conda activate C3PO

# Install PyTorch (for CUDA 12.3)
conda install pytorch torchvision torchaudio pytorch-cuda=12.3 -c pytorch -c nvidia -y

# Install required packages
pip install torch numpy transformers fvcore tqdm
```

### 2. Download Reference Cases

Download the reference cases from this link:
[Reference Cases](https://drive.google.com/file/d/1hw3nW7b8hG0KkL0C3kDUZ8Pkk2ywzv-f/view?usp=sharing)

```bash
# Extract the downloaded reference.zip
unzip reference.zip -d reference_data
```

### 3. Download Datasets

Run the `download.sh` script to get the necessary datasets:

```bash
# Execute download script
bash download.sh
```

### 4. Replace the original olmoe_modeling.py to customize the routing weights

```bash
# Replace olmoe_modeling.py
cp olmoe_modeling.py /miniconda3/envs/C3PO/lib/python3.10/site-packages/transformers/models/olmoe/modeling_olmoe.py
```

### 5. Run the optimizer code

```bash
# Run the main script
python olmoe_optimizer.py
```
