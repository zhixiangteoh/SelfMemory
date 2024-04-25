# Efficient and Precise Self-Memory for RAG

[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![transformers](https://img.shields.io/badge/Transformers-orange)](https://github.com/huggingface/transformers)


This repository is a public fork based on the source code for the **Selfmem** framework presented in this paper [Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory](https://arxiv.org/abs/2305.02437).

With direct access to human-written reference as memory, retrieval-augmented generation has achieved much progress in a wide range of text generation tasks. Since better memory would typically prompt better generation (we define this as __primal problem__), previous works mainly focus on how to retrieve better memory. **Selfmem** addresses a fundamental limitation that exists for current literature: the memory is retrieved from a fixed corpus and is bounded by the quality of the corpus. In this paper, by exploring the __duality__ of the primal problem: better generation also prompts better memory, the **Selfmem** framework iteratively adopts a retrieval-augmented generator itself to generate an unbounded memory pool and uses a memory selector to pick one generated memory for the next generation round. By combining the primal and dual problem, a retrieval-augmented generation model could **lift itself up** with its own output in the infinite generation space.

![original-selfmem-framework](model.svg)
*Figure 1: Overall Selfmem framework diagram, taken from Figure 2 of the original paper. There are two components in Selfmem, a retrieval-augmented generator (a) and a memory selector (b). For the primal problem, (a) takes source and memory as input to generate candidates for (b). For the dual problem, (b) takes as input source and generated candidates to select memory for (a).*


We build on top of this idea by implementing the following two optimizations (see Figure 2):
1) **Top-B Memory Beam Search**:  Memory selector currently explores top-1 candidate. We replace this with a top-B algorithm implemented with data parallelization across multiple GPUs on multiple machines to decrease the iterations to convergence.
2) **Augmenting generator memory**: Instead of using the one best candidate for each training instance to replace the existing memory, we integrate the memory selector’s top candidate upon each iteration for a more comprehensive set of memory to aid in generation quality.

![approaches-augmented-diagram](approaches-augmented-diagram.jpg)
*Figure 2: Original Selfmem framework diagram modified with our two optimizations. Top-B memory beam search optimization is implemented on top of the memory selector (b), while augmenting generator memory optimization is implemented on top of the retrieval-augmented generator (a).*

Additionally, we enhance the codebase by cleaning up the original **Selfmem** repository, adding scripts for the main iterative generation / memory selection process, and generalizing the code to use publicly available Huggingface pretrained models, both for the retrieval-augmented generator (`BartForConditionalGeneration`) and the memory selector reranker (`BGE-M3`).

---

## Setup
This code is mainly based on ⚡ [PyTorch Lightning]() and 🤗 [Transformers](https://github.com/huggingface/transformers). 

Specifically, the model definition and tokenizer is based on 🤗, and the Trainer is from ⚡.

The project is run on multiple c240g5 nodes in a pytorch cluster on [CloudLab](https://docs.cloudlab.us/cloudlab-manual.html). Each node is equipped with one NVIDIA 12GB PCI P100 GPU.

The following set-up steps are meant to be run on each node within the cluster:

1. Create disk partition for extra space
```bash
sudo mkdir /mydata
sudo /usr/local/etc/emulab/mkextrafs.pl /mydata
df -h # Should see /mydata
sudo chown -R <user> /mydata
```

2. Install CUDA toolkit and drivers:
```bash
mkdir /mydata/cuda
mkdir /mydata/cuda/bin
cd /mydata/cuda
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo bash cuda_11.7.0_515.43.04_linux.run --toolkitpath=/mydata/cuda/bin/cuda-11.7
# Accept and leave all installs selected
# If there is an issue about Nouveau in /var/log/nvidia-installer.log run these commands:
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
# Insert these 2 lines:
blacklist nouveau 
options nouveau modeset=0

sudo update-initramfs -u
sudo reboot

# After reboot ssh again
export PATH="/usr/bin:$PATH"
export PATH="/mydata/cuda/bin/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/mydata/cuda/bin/cuda-11.7/lib64"
export CUDA_HOME="/mydata/cuda/bin/cuda-11.7"
source ~/.bashrc

cd /mydata/cuda
sudo bash cuda_11.7.0_515.43.04_linux.run --toolkitpath=/mydata/cuda/bin/cuda-11.7
nvidia-smi   # should have output
```

3. Install Python 3.10.8
```bash
sudo apt update && sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

mkdir /mydata/pyenv
git clone https://github.com/pyenv/pyenv.git /mydata/pyenv
export PYENV_ROOT="/mydata/pyenv"
sudo git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
source ~/.bashrc

pyenv init
pyenv install 3.10.8
pyenv global 3.10.8
python --version   #Should be 3.10.8

mkdir /mydata/tmp
mkdir /mydata/pip_cache
export TMPDIR=/mydata/tmp
export PIP_CACHE_DIR=/mydata/pip_cache
source ~/.bashrc
```

4. Clone this repo and install Python package requirements
```bash
cd /mydata
git clone git@github.com:zhixiangteoh/SelfMemory.git
cd SelfMemory
pip install --cache-dir=/mydata/pip_cache -r requirements.txt
```

5. Verify Pytorch and CUDA are correctly installed
```python
>>> import torch
>>> print(torch.cuda.is_available())
True
>>> print(torch.cuda.get_device_name(0))
Tesla P100-PCIE-12GB
```

6. Set up the data directory, first downloading and preparing the corresponding dataset as described in the "Data" section. It should contain minimally the following files: `src.vocab`, `tgt.vocab`, `test.jsonl`, `test_src.txt`, `test_trg.txt`, `test.txt`, and a `memory` sub-directory with `random.memory` file.

```bash
# Example DATA_DIR="/mydata/SelfMemory/data/ende" set-up:
tree /mydata/SelfMemory/data
# /mydata/SelfMemory/data
# ├── ende
# │   ├── memory
# │   │   ├── random.memory
# │   ├── src.vocab
# │   ├── test.jsonl
# │   ├── test_src.txt
# │   ├── test_trg.txt
# │   ├── test.txt
# │   └── tgt.vocab
```

---
## Data
For dataset we use:
- **JRC-Acquis**. We use the data version from this [paper](https://aclanthology.org/2021.acl-long.567.pdf). For downloading, we refer to this [LINK](https://drive.google.com/file/d/1iuBH_YsnL28cTYjjpSq5BgukG7QhBLs_/view) to download the data and this [script](https://github.com/jcyk/copyisallyouneed/blob/master/scripts/prepare.sh) for data pre-processing.

- **XSum** is downloaded from this [LINK](https://github.com/yixinL7/BRIO).

After downloading the data, we make it in the format of `Jsonline`  and put it in the `data` folder.

For initial memory retrieval, we use `ElasticSearch` to conduct first-stage memory retrieval based on BM25 score.

We also provide the final hypothesis and reference in the [output](output) directory for potential need. For evaluation scripts, please refer to [metrics_utils.py](src/utils/metrics_utils.py)

---

## Running

Run script for Selfmem main loop, specifiying minimally an experiment directory (`EXPNAME`) to save output to, and optionally a data directory (`DATA_DIR`) that contains the inference dataset. By default, `DATA_DIR` is set to `/mydata/SelfMemory/data/ende` for an EN-DE translation task on the JRC-Acquis EN-DE dataset.

```bash
cd /mydata/SelfMemory/src

./selfmem.sh EXPNAME="baseline" DATA_DIR="/mydata/SelfMemory/data/ende"
```

---

## SelfMem components

The following sections describe the main components of the SelfMem framework, including the retrieval-augmented generator and memory selector, and how to train and evaluate them. One of our main contributions is implementing the pseudo-code for the iterative generation/memory selection process.

### Retrieval-Augmented Generator
Here we use JRC-Acqius EnDe dataset as example:
```bash
cd your/work/dir/src

## train a vanilla Transformer model
python train_generator.py \
    --config config/jrc_ende/train_generator.yaml \
    --precision 16

## Transformer-Joint
python train_generator.py \
    --config config/jrc_ende/train_generator.yaml \
    --precision 16 \
    --memory_encoding concate \
    --memory_dir ../data/jrc_ende/memory/bm25 

## Transformer-Dual
python train_generator.py \
    --config config/jrc_ende/train_generator.yaml \
    --precision 16 \
    --memory_encoding separate \
    --memory_dir ../data/jrc_ende/memory/bm25 
```

### Memory Selector
First we use the trained generator to generate candidates
```bash
cd your/work/dir/src

python generate_hyps.py \
	--config config/jrc_ende/generate_hyps.yaml \
    --num_beams 50 --num_return_sequences 50 \
	--data_path ../data/jrc_ende/test.jsonl \
	--pretrained_model_path your/trained/model/path
	--memory_encoding concate \
	--memory_path ../data/jrc_ende/memory/bm25/test.txt \
	--output_path output.txt
```
Then we using the script in /src/select_mem.py to train the memory selector.

## Lift Yourself Up
Here is the pseudo code for the whole process:
```python
generator = Trainer(model_input,memory)
candidates = generator(model_input,memory)
selector = Trainer(model_input,candidates)

for _ in range(iteration_k):
    candidates = generator(model_input,memory)
    memory = selector(model_input,candidates)
    hypothesis = generator(model_input,memory)
    current_score = metrics(hypothesis,reference)
```
