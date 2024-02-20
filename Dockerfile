FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

WORKDIR /home

# Install required apt packages
RUN apt update -y --fix-missing
RUN apt install -y byobu git wget unzip htop zsh parallel

# Check if an NVIDIA GPU is available
RUN wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda3
RUN rm ./Miniconda3-latest-Linux-x86_64.sh
RUN ln -s /home/miniconda3/bin/conda /usr/bin/conda

RUN conda update -n base -c defaults conda -y
RUN conda init
ENV PATH /home/miniconda3/bin:$PATH
RUN conda install python=3.10 -y
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
RUN conda install pip sentencepiece pydantic python-dotenv black isort tqdm wandb pandas matplotlib accelerate scikit-learn hydra-core pynvml -c conda-forge -y
RUN conda install transformers datasets huggingface_hub evaluate -c huggingface -y
RUN rm -rf /home/miniconda3/pkgs/*
RUN PYTHONDONTWRITEBYTECODE=1
RUN pip install deepspeed
RUN pip install peft
RUN pip install -U pydantic==1.10.12
RUN pip install -U transformers
RUN pip install -U tokenizers
RUN pip install bitsandbytes
RUN pip install scispacy
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
RUN pip install rank_bm25
RUN pip install nltk
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords