# 🚀 Data Parallel Inference Framework

## 🌟 Overview
This project delivers a sophisticated data parallel inference framework leveraging PyTorch and the LLaMA 3.2-3B model, optimized for distributed processing across four GPU-enabled nodes. Integrated with SLURM for seamless job orchestration, it ensures high-throughput inference with minimal latency. The codebase is crafted for scalability, reliability, and elegance in high-performance computing environments.

## ✨ Features

- 🖥️ Distributed Inference: Utilizes PyTorch's distributed module with NCCL backend for efficient GPU communication.
- ⚙️ SLURM Integration: Orchestrates jobs across nodes for optimal resource allocation.
- 💾 Efficient Model Loading: Loads LLaMA 3.2-3B in FP16 precision to minimize memory usage.
- 📊 Dynamic Prompt Distribution: Automatically balances prompt workloads across nodes.
- 📈 Comprehensive Metrics: Tracks token generation speed, processing time, and GPU memory usage.

## 📁 Repository Structure

- 📜 Data_Parallel_4_Nodes.py: Core script for data parallel inference with PyTorch and Transformers.
- 🔧 Inference.slurm: SLURM script for configuring and running distributed inference tasks.
- 📖 README.md: This file, your guide to the project.

## 🛠️ Prerequisites

### Hardware:
- 🖥️ 4 GPU-enabled nodes (NVIDIA GPUs with CUDA 12.2 support).
### Software:
- 🐍 Python 3.9+
- 🔥 PyTorch with CUDA support
- 🤗 Hugging Face Transformers
- 📡 SLURM workload manager
- 🔗 NCCL for distributed GPU communication

## Model:

📂 LLaMA 3.2-3B model weights at /mnt/lustre/user/llama3.2-3b-instruct.

## ⚙️ Setup Instructions

### Clone the Repository:

- git clone <repository-url>
- cd <repository-directory>

### Set Up Environment: 
Activate the Python environment and install dependencies:

- module load python/3.9 cuda/12.2
- source /mnt/lustre/user/llm-env/bin/activate
- pip install torch transformers

### Configure SLURM: 
Update NCCL_SOCKET_IFNAME in Inference.slurm to match your cluster's network interfaces.

### Prepare Model: 
Ensure LLaMA 3.2-3B model weights are available at the specified MODEL_PATH.

### 🚀 Running the Framework

- Launch the distributed inference job with SLURM:
- sbatch Inference.slurm

### What Happens?

- 🔗 Initializes the distributed environment with NCCL.
- 📥 Loads the LLaMA 3.2-3B model on each node.
- 📤 Distributes prompts across nodes for parallel processing.
- 📊 Collects and displays performance metrics.

## 📜 Output
Results are logged to:
- 📄 data_parallel_test_%j.out (output logs)
- ⚠️ data_parallel_test_%j.err (error logs)
The master node (rank 0) aggregates results, displaying per-node and overall performance metrics, including token generation rates and processing times.

## ⚡ Performance Optimization

### 🔌 NCCL Configuration:
Tune NCCL_SOCKET_IFNAME and set NCCL_IB_DISABLE=1 for network compatibility.



💾 Memory Management: Uses FP16 to fit models on GPUs with ≥
