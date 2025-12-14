# Accelerators

CuEquivariance and flashTP provide acceleration for both SevenNet training and inference. For inference speed benchmark results, follow [here](https://arxiv.org/abs/2510.11241)

## CuEquivariance

CuEquivariance is an NVIDIA Python library designed to facilitate the construction of high-performance geometric neural networks using segmented polynomials and triangular operations. CuEquivariance accelerates SevenNet during training, inference with ASE and LAMMPS via ML-IAP. For more information, refer to [cuEquivariance](https://github.com/NVIDIA/cuEquivariance).

### Requirements
- Python >= 3.10
- cuEquivariance >= 0.6.1

### Installation
Install via:

```bash
pip install sevenn[cueq12]  # cueq11 for CUDA version 11.*
```

:::{note}
Some GeForce GPUs do not support `pynvml`,
causing `pynvml.NVMLError_NotSupported: Not Supported`.
Then try a lower cuEquivariance version, such as 0.6.1.
:::

## FlashTP

FlashTP, presented in [FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials](https://openreview.net/forum?id=wiQe95BPaB), is a high-performance Tensor-Product library for Machine Learning Interatomic Potentials (MLIPs). FlashTP accelerates SevenNet during training and inference with ASE, LAMMPS (single & multi-GPU), LAMMPS via ML-IAP. For more information, refer to [flashTP](https://github.com/SNU-ARC/flashTP).

### Requirements
- Python >= 3.10
- flashTP >= 0.1.0
- CUDA toolkit >= 12

### Installation
Install via:
```bash
git clone https://github.com/SNU-ARC/flashTP.git
cd flashTP
pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
```
Customize CUDA_ARCH_LIST to match the [compute compatibility](https://developer.nvidia.com/cuda/gpus) of the GPU.


:::{note}
During installation of flashTP,
`subprocess.CalledProcessError: ninja ... exit status 137`
typically indicates **out-of-memory** during compilation.
Try reducing the build parallelism:
```bash
export MAX_JOBS=1
```
:::

## CuEquivariance, flashTP usage

CuEquivariance and FlashTP can be used with:

| Feature | Training | ASE | LAMMPS | LAMMPS via ML-IAP |
|--------------------|----------|-----|--------|--------|
| **flashTP** | [Training](./cli.md#sevenn-train) | [ASE](./ase_calculator.md) | [LAMMPS (single & multi-GPU)](./lammps_torch.md#build) | [ML-IAP](./lammps_mliap.md#potential-deployment) |
| **cuEq** | [Training](./cli.md#sevenn-train) | [ASE](./ase_calculator.md) | — | [ML-IAP](./lammps_mliap.md#potential-deployment) |

:::{caution}
Currently, among the available accelerators, only **flashTP without D3(./d3.md)** supports multi-GPU LAMMPS.
:::
