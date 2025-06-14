# CS336 Spring 2025 Assignment 1: Basics
CS336 Assignment 1 的个人实现，包括一个 bpe 分词器和一个 Transformer 模型，以及相应的训练和推理（解码）流程。有较多的中文注释。
仅供参考。如果对你有帮助欢迎加个 star，如有相关问题需要探讨，也可联系 hannah976@qq.com
## 目录结构
主要代码在 `./cs336_basics` 下，包含以下模块：
```text
cs336_basics
├── bpe_tokenizer     # 基于字节对编码(BPE)的分词器实现
│   ├── pre_tokenizer.py  # 预分词器
│   ├── tokenizer.py      # BPE编解码实现
│   └── trainer.py        # BPE训练器实现和训练脚本
└── transformer       # Transformer语言模型实现
    ├── module.py         # 模型架构（transformer及其各个模块）
    ├── trainer_utils.py  # 训练相关工具（loss、优化器等）
    ├── train.py          # 模型训练脚本
    └── inference.py      # 推理及解码脚本
```
原 README
---

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
````

