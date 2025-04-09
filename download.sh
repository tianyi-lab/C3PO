#!/bin/bash

# Install Git LFS
git lfs install

# Create directory for reference data if it doesn't exist
mkdir -p reference_data

# Clone datasets from Hugging Face
git clone https://huggingface.co/datasets/allenai/sciq
git clone https://huggingface.co/datasets/allenai/openbookqa
git clone https://huggingface.co/datasets/allenai/ai2_arc

echo "All datasets downloaded successfully!"
