<div align="center">

# CliffordNet: All You Need is Geometric Algebra
  


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Github](https://img.shields.io/badge/Github-grey?logo=github)](https://github.com)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.06793-b31b1b.svg)](https://arxiv.org/abs/2601.06793)
[![Hardware](https://img.shields.io/badge/Triton-Accelerated-blue)](https://triton-lang.org/)

“The two systems [Hamilton’s and Grassmann’s]
are not only consistent with one another, but they
are actually parts of a larger whole.”

— William Kingdon Clifford, 1878

</div>

Official implementation of the paper **"CliffordNet: All You Need is Geometric Algebra"**.

We introduce **Clifford Algebra Network (CAN)**, a novel vision backbone that challenges the necessity of Feed-Forward Networks (FFNs) in deep learning. By operationalizing the full **Clifford Geometric Product** ($uv = u \cdot v + u \wedge v$), we unify feature coherence and structural variation into a single, algebraically complete interaction layer.

Our **"No-FFN"** variant demonstrates that this geometric interaction is so expressive that heavy MLPs become redundant, establishing a new Pareto frontier for efficient visual representation learning.

## 🚀 News & Updates

*   **[2026-01-12]** ⚡ **Performance Preview:** We have successfully implemented a custom **Fused Triton Kernel** for the Clifford Interaction layer. Preliminary benchmarks on RTX 4090 show a **10x kernel speedup** and **~2x end-to-end training speedup**. *Code coming soon!*
*   **[2026-01-1]** 🏆 **SOTA on CIFAR-100:** Our Nano model (1.4M) matches ResNet-18 (11M), and our No-FFN model outperforms MobileNetV2 by >6%.
