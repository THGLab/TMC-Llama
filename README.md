# TMC-Llama: Exploring Transition Metal Complexes with Large Language Models

[![License](https://img.shields.io/github/license/THGLab/TMC-Llama)](LICENSE)  
[DOI](10.26434/chemrxiv-2025-hm3zb)  
[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/69136d39a10c9f5ca1c14847)


## 📖 Introduction

TMC-Llama is a language model fine-tuned from Meta's open-source Llama3 (Llama-3.2-1b-Instruct) for generating **transition metal complexes (TMCs)** in SMILES notation. It uses [TMC-SMILES](https://doi.org/10.1186/s13321-025-01008-1) (Rasmussen et al.), a format designed for RDKit-compatible metal–organic structures. Given target chemical properties in the supervised fine-tuning (SFT) prompts, TMC-Llama generates TMCs in desired chemical regions, supporting discovery and screening workflows.

The accompanying paper analyzes **unparsable SMILES** (see Notebook 2) and describes failure modes of generated TMCs. We link these failure modes to molecular features and use them to improve SFT protocols and post-generation corrections. These insights can support future tools for generating chemically valid TMCs.

## 🔍 How to Use

### 📕 Llama3 Environment

TMC-Llama inference requires **PyTorch**, **Transformers**, and **RDKit**:

- **PyTorch**: [torch](https://pytorch.org/get-started/previous-versions/). TMC-Llama uses [CUDA](https://developer.nvidia.com/cuda/toolkit) (version 11.8) to run PyTorch.
- **Transformers**: [Huggingface transformers](https://huggingface.co/docs/transformers/en/installation). You can set custom `CACHE` directories if needed.
- **RDKit**: [RDKit](https://www.rdkit.org/docs/Install.html)

Inference utilities are in `libllama/`, adapted from the [SmileyLlama](https://github.com/THGLab/SmileyLlama) project. The virtual environment setup matches SmileyLlama.

### 📗 Running Jupyter Notebook Demonstrations

The notebooks rely on code in `libTMC/` and `libllama/`. Make sure RDKit and the other prerequisites above are installed. `libTMC/` provides Python utilities for:

- Detecting transition metal centers
- Extracting ligands
- Fixing redundant dative bonds
- Correcting improper valences and unclosed rings
- Parse TMC-SMILES, redirect I/O streams, and identify errors

Example datasets and outputs (`.csv` files) are in the `data/` directory.
Inference notebook 4 generate SMILES strings in example text format (such as the `example.txt` in `txt/`). Cleaned TMC-SMILES (removing identical strings) and parsability errors are in `E_*.csv` and `B_*.csv` files in `par/`.

### 📘 Fine-Tuning TMC-Llama

TMC-Llama is built on [SmileyLlama](https://github.com/THGLab/SmileyLlama). Install **axolotl** following the [Installation guide](https://github.com/THGLab/SmileyLlama/tree/main?tab=readme-ov-file#installation-guide). The fine-tuning dataset and SFT prompts are available on [FigShare](https://doi.org/10.6084/m9.figshare.30594500).

### 📙 Inference

To run inference:

1. Download the trained models from [FigShare](https://doi.org/10.6084/m9.figshare.30594500)
2. Follow the instructions in **Notebook 4** (inference guideline)

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank all contributors who developed TMC-Llama and built this project. Related Llama3 applications for chemistry are available in [SmileyLlama](https://github.com/THGLab/SmileyLlama) and [SynLlama](https://github.com/THGLab/SynLlama/).

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{tmc_llama_2025,
    title = {Exploring Transition Metal Complexes with Large Language Models},  
    url = {https://chemrxiv.org/engage/chemrxiv/article-details/69136d39a10c9f5ca1c14847},
    doi = {10.26434/chemrxiv-2025-hm3zb},
    publisher = {ChemRxiv},
    author = {Liu, Yunsheng and Cavanagh, Joseph and Sun, Kunyang and Toney, Jacob and Yuan, Chung-Yueh and Smith, Andrew and St Michel II, Roland and Graggs, Paul and Toste, F Dean and Kulik, Heather and Head-Gordon, Teresa},
    month = nov,
    year = {2025}}
```
