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
*   **[2026-01-01]** 🏆 **SOTA on CIFAR-100:** Our Nano model (1.4M) matches ResNet-18 (11M), and our No-FFN model outperforms MobileNetV2 by >6%.

## 🏆 Main Results (CIFAR-100)

We compare CliffordNet against established efficient backbones under a rigorous "Modern Training Recipe" (200 Epochs, AdamW, AutoAugment, DropPath).

| Model Variant          |  Params  |  FFN?   | Resolution | Top-1 Acc  | vs. Baseline                                                 |
| :--------------------- | :------: | :-----: | :--------: | :--------: | :----------------------------------------------------------- |
| **Baselines**          |          |         |            |            |                                                              |
| MobileNetV2            |   2.3M   |   Yes   |   $32^2$   |   70.90%   | -                                                            |
| ViT-Tiny               |   5.7M   |   Yes   |   $32^2$   |   72.50%   | -                                                            |
| ResNet-18              |  11.2M   |   Yes   |   $32^2$   |   76.63%   | -                                                            |
| **CliffordNet (Ours)** |          |         |            |            |                                                              |
| **CAN-Nano**           | **1.4M** | **No**  |   $32^2$   | **76.41%** | <span style="color:green">**Match ResNet-18 (1/8 Params)**</span> |
| **CAN-Fast**           | **2.6M** | **No**  |   $32^2$   | **77.63%** | <span style="color:green">**+6.7% vs MobileNet**</span>      |
| **CAN-Base**           |   3.0M   | Ratio=1 |   $32^2$   | **78.05%** | <span style="color:green">**SOTA**</span>                    |

> **Key Insight:** The **CAN-Fast** model completely removes the FFN block (mlp_ratio=0), yet achieves **77.63%** accuracy. This empirically validates that **geometric interactions > generic depth**.

## 🏗️ Architecture

The core of CliffordNet is the **Dual-Stream Geometric Block**, governed by the discretized geometric evolution equation:

$$ 
\frac{\partial H}{\partial t} = \mathcal{P}\Big( \underbrace{H \cdot \mathcal{C}}_{\text{Diffusion}} \oplus \underbrace{H \wedge \mathcal{C}}_{\text{Geometric Flow}} \Big) 
$$

Where $\mathcal{C}$ is the local context approximated by a **Factorized Linear Laplacian**, and the interaction is fused via our **Gated Geometric Residual (GGR)** mechanism.

## 🖊️ Citation

If you find this work helpful, please cite us:

```bibtex
@article{2026cliffordnet,
  title={CliffordNet: All You Need is Geometric Algebra},
  author={Zhongping Ji},
  journal={arXiv preprint arXiv:2601.06793},
  year={2026}
}
```

## 🙏 Acknowledgement

We thank the open-source community for the implementations of `timm`, which facilitated our baseline comparisons. 

