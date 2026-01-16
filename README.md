# TMC-Llama: Exploring Transition Metal Complexes with Large Language Models

[![License](https://img.shields.io/github/license/THGLab/TMC-Llama)](LICENSE)  
[DOI](10.26434/chemrxiv-2025-hm3zb)  
[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/69136d39a10c9f5ca1c14847)


## 📖 Introduction
TMC-Llama is fine-tuned from Meta's open-source pre-trained Llama3 large language models (Llama-3.2-1b-Instruct). TMC-Llama generates transition metal complexes (TMCs) using SMILES notations that are tailored for RDKit-compatible metal-organic connections, [TMC-SMILES](https://doi.org/10.1186/s13321-025-01008-1) (developed by Rasmussen and co-workers). With a set of chemical properties provided in the supervised fine-tuning (SFT) prompts, TMC-Llama can generate TMCs in specific chemical space, making TMC-Llama a useful tool for TMC discovery.
  
In addition, the paper studies the unparsable strings (in Notebook 2) and identifies several failure modes for the generated TMCs. Corresponding to these failure modes, we revealed characteristic molecular properties / features that are helpful to build future tools, including SFT protocols and post-generation algorithms, for high quality TMC generation. These properties can also be infrastructures to develop models for chemically functional TMCs.

## 🔍 How to use

### 📕 Llama3 environment
Performing inference for TMC-Llama only requires installation of `PyTorch`, `Transformers`, and `RDKit`, which can be found in the directories below:
- PyTorch: [torch](https://pytorch.org/get-started/previous-versions/). In addition, TMC-Llama utilizes [CUDA](https://developer.nvidia.com/cuda/toolkit) (version 11.8) to run `PyTorch`.
- Transformers: [Huggingface transformers](https://huggingface.co/docs/transformers/en/installation). Note that you may want to specify your preferred `CACHE` directories.
- RDKit: [RDKit](https://www.rdkit.org/docs/Install.html)  

All customized `.py` files to perform inference are in the `libllama/` directory, which are developed in [SmileyLlama](https://github.com/THGLab/SmileyLlama) project. The prerequisites of virtual environment to perform inference will be identical to SmileyLlama as well.

### 📗 Running Jupyter-notebook demonstrations
All notebook demonstrations can be performed using existing files in `libTMC/` and `libllama/` directories if the prerequisites above are satisfied, such as `RDKit`. Customized python functions to identify transition metal centers, isolate ligands, fix redundant dative bonds, correct atoms with improper valences, and fix unclosed rings are in `.py` files in `libTMC/`.  
Demonstration datasets and the generated results (both of which are `.csv` files) are in `data/` directory.

### 📘 Fine-tuning TMC-Llama
TMC-Llama is built on top of the SmileyLlama repository, so `axolotl` needs to be installed to fine-tune and obtain TMC-Llama, following the previous [Installation guide](https://github.com/THGLab/SmileyLlama/tree/main?tab=readme-ov-file#installation-guide). The fine-tuning dataset of TMC-Llama and the corresponding SFT prompts can be found on [FigShare]().

### 📙 Inference
To perform inference using TMC-Llama, download the trained models from [FigShare]() and follow the instructions in the Notebook 4 (inference guideline).

## 📄 License
See the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments
We thank all authors to develop TMC-Llama and build this project! Similar utility of Llama3 models for bio-chemical applications can be found in [SmileyLlama](https://github.com/THGLab/SmileyLlama) and [SynLlama](https://github.com/THGLab/SynLlama/).

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
