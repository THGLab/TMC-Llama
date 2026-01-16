# TMC-Llama: Exploring Transition Metal Complexes with Large Language Models

[![License](https://img.shields.io/github/license/THGLab/TMC-Llama)](LICENSE)
  
[DOI](10.26434/chemrxiv-2025-hm3zb)
  
[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/69136d39a10c9f5ca1c14847)


## 📖 Introduction
TMC-Llama is fine-tuned from Meta's open-source pre-trained Llama3 large language models (Llama-3.2-1b-Instruct). TMC-Llama generates transition metal complexes (TMCs) using SMILES notations that are tailored for RDKit-compatible metal-organic connections, [TMC-SMILES](https://doi.org/10.1186/s13321-025-01008-1) (developed by Rasmussen and co-workers). With a set of chemical properties provided in the supervised fine-tuning (SFT) prompts, TMC-Llama can generate TMCs in specific chemical space, making TMC-Llama a useful tool for TMC discovery.
  
In addition, the paper studies the unparsable strings (in Notebook 2) and identifies several failure modes for the generated TMCs. Corresponding to these failure modes, we revealed characteristic molecular properties / features that are helpful to build future tools, including SFT protocols and post-generation algorithms, for high quality TMCs. These properties can also be infrastructures to develop models for chemically functional TMCs.

## 💡 Usage

### Prerequisites
.

### Installation
.

### Inference
To perform inference using the already trained TMC-Llama, download the trained models and relevant files from [here]() and follow the instructions in the [Inference Notebook]().

## 📄 License
See the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments


## 📝 Citation
If you use this code in your research, please cite:

```bibtex
@misc{tmc_llama_2025,
    title = {},  
    url = {},
    doi = {},
    publisher = {},
    author = {},
    month = nov,
    year = {2025}
