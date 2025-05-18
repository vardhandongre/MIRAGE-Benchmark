# MIRAGE Benchmark

<div align="center">
  <img src="assets/data-logo.png" alt="MIRAGE Data Logo" width="300"/>
</div>

## Overview

MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agricultural Expert-Guided Conversations is a comprehensive benchmark for evaluating AI systems in multimodal consultation scenarios. The benchmark consists of two main components:

- [MMST (Multi-Modal Single-Turn)](MMST/README.md) - Single-turn multimodal reasoning tasks
- [MMMT (Multi-Modal Multi-Turn)](MMMT/README.md) - Multi-turn conversational tasks with visual context

## Dataset

The full benchmark dataset is available on Hugging Face:
[MIRAGE-Benchmark/MIRAGE](https://huggingface.co/MIRAGE-Benchmark)

```python
from datasets import load_dataset

# Load MMST datasets
ds_standard = load_dataset("MIRAGE-Benchmark/MIRAGE", "MMST_Standard")
ds_contextual = load_dataset("MIRAGE-Benchmark/MIRAGE", "MMST_Contextual")

# Load MMMT dataset
ds_mmmt_direct = load_dataset("MIRAGE-Benchmark/MIRAGE", "MMMT_Direct")
ds_mmmt_decomp = load_dataset("MIRAGE-Benchmark/MIRAGE", "MMMT_Decomp")
```

## Paper

Our paper is available on arXiv:
[MIRAGE](https://arxiv.org/abs/XXXX.XXXXX)

## Citation

If you use our benchmark in your research, please cite our paper:

```bibtex
@article{mirage2024,
  title={MIRAGE},
  author={Dongre, Vardhan and Gui, Chi and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0).

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

See the [LICENSE](LICENSE) file for details.
