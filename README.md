# C3PO: Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

Mixture-of-Experts (MoE) Large Language Models (LLMs) suffer from severely sub-optimal expert pathways—our study reveals that naive expert selection learned from pretraining leaves a surprising 10-20% accuracy gap for improvement. Motivated by this observation, we develop a novel class of test-time optimization methods to re-weight or “re-mixing” the experts in different layers jointly for each test sample. Since the test sample’s ground truth is unknown, we propose to optimize a surrogate objective defined by the sample’s “successful neighbors” from a reference set of samples. We introduce three surrogates and algorithms based on mode-finding, kernel regression, and the average loss of similar reference samples/tasks. To reduce the cost of optimizing whole pathways, we apply our algorithms merely to the core experts’ mixing weights in critical layers, which enjoy similar performance but save significant computation. This leads to “Critical-Layer, Core-Expert, Collaborative Pathway Optimization (C3PO)”. We apply C3PO to two recent MoE LLMs and examine it on six widely-used benchmarks. It consistently improves the base model by 7-15% in accuracy and outperforms widely used test-time learning baselines, e.g., in-context learning and prompt/prefix tuning, by a large margin. Moreover, C3PO enables MoE LLMs with 1-3B active parameters to outperform LLMs of 7-9B parameters, hence improving MoE’s advantages on efficiency. Our thorough ablation study further sheds novel insights on achieving test-time improvement on MoE.

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
