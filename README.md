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

*   **[2026-02-17]** 🔥 **Code Release:** Model and training code will be released!
*   **[2026-01-20]** 🏆 **New SOTA:**
    *   **Nano (1.4M)** reaches **77.82%**, outperforming ResNet-18 (11M).
    *   **Lite (2.6M)** reaches **79.05%** without FFN, rivaling ResNet-50.
    *   **32-Layer Deep Model** achieves **81.42%** with only 4.8M parameters.
*   **[2026-01-12]** ⚡ **Performance Preview:** We have successfully implemented a custom **Fused Triton Kernel** for the Clifford Interaction layer. Preliminary benchmarks on RTX 4090 show a **10x kernel speedup** and **~2x end-to-end training speedup**. *Code coming soon!*
*   **[2026-01-01]** 🏆 **SOTA on CIFAR-100:** Our Nano model (1.4M) matches ResNet-18 (11M), and our No-FFN model outperforms MobileNetV2 by >6%.

## 🏆 Main Results (CIFAR-100)

We compare CliffordNet against established efficient backbones under a rigorous "Modern Training Recipe" (200 Epochs, AdamW, AutoAugment, DropPath).

### Efficiency & Performance
| Model Variant | Params | MLP Ratio | Context Mode | Top-1 Acc | vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baselines** | | | | | |
| MobileNetV2 | 2.3M | - | - | 70.90% | - |
| ShuffleNetV2 1.5x | 2.6M | - | - | 75.95% | - |
| ResNet-18 | 11.2M | - | - | 76.75% | - |
| ResNet-50 | 23.7M | - | - | 79.14% | - |
| **CliffordNet (Ours)** | | | | | |
| **CAN-Nano** | **1.4M** | **0.0** | Diff ($\Delta H$) | **77.82%** | <span style="color:green">**> ResNet-18**</span> |
| **CAN-Lite** | **2.6M** | **0.0** | Diff ($\Delta H$) | **79.05%** | <span style="color:green">**~ ResNet-50**</span> |
| **CAN-32 (Deep)**| 4.8M | 0.0 | Full | **81.42%** | <span style="color:green">**SOTA**</span> |
| **CAN-64 (Deep)**| 8.6M | 0.0 | Full | **82.46%** | <span style="color:green">**SOTA**</span> |

> **Key Insight:** Our **Nano** variant (1.4M) outperforms the heavy-weight **ResNet-18** (11.2M) by **+1.07%** while using **$8\times$ fewer parameters**. The **Lite** variant (No-FFN) effectively matches ResNet-50 with **$9\times$ fewer parameters**.

## 🏗️ Architecture

The core of CliffordNet is the **Dual-Stream Geometric Block**, governed by the dynamic geometric evolution equation:

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

