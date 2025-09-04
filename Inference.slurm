#!/bin/bash
#SBATCH --job-name=data_parallel_test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=data_parallel_test_%j.out
#SBATCH --error=data_parallel_test_%j.err

# Environment setup
module load python/3.9
module load cuda/12.2
source /mnt/lustre/user/llm-env/bin/activate

# Basic config
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

# Master node setup
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# NCCL settings - cleaner setup
export NCCL_IB_DISABLE=1  # Disable InfiniBand
export NCCL_SOCKET_IFNAME=ens1f1  # Default interface

# Reduce NCCL verbosity - only show warnings/errors
export NCCL_DEBUG=WARN  # Changed from INFO to WARN

# Disable problematic plugins to avoid warnings
export NCCL_NET_PLUGIN=none
export NCCL_TUNER_PLUGIN=none

# Optional: Specify network path explicitly
export NCCL_NET="Socket"

# Debug output
export PYTHONUNBUFFERED=1

echo "========================================"
echo "DATA PARALLEL INFERENCE TEST"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Master node: $MASTER_ADDR"
echo "========================================"

# Display node info
echo "Node information:"
srun --ntasks=4 --ntasks-per-node=1 bash -c '
    echo "[$(hostname)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "[$(hostname)] Memory: $(free -h | grep Mem | awk "{print \$2}")"
    echo ""
'

echo "========================================"
echo "Starting data parallel inference..."
echo "========================================"

# Run with rank and world size environment variables
srun bash -c '
# Set distributed environment variables
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export SLURM_LOCALID=$SLURM_LOCALID

# Set network interface per node (simplified)
case $(hostname) in
  master-node) export NCCL_SOCKET_IFNAME=ens1f1 ;;
  node01) export NCCL_SOCKET_IFNAME=ens1f1 ;;
  node02) export NCCL_SOCKET_IFNAME=ens1f0np0 ;;
  node03) export NCCL_SOCKET_IFNAME=ens1f1 ;;
  *) export NCCL_SOCKET_IFNAME=ens1f1 ;;
esac

echo "[$(hostname):R$RANK] Starting with interface $NCCL_SOCKET_IFNAME"

# Run the data parallel test
python data_parallel_4node.py
'

echo "========================================"
echo "Data parallel test completed at $(date)"
echo "========================================"