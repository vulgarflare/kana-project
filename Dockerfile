FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN apt update && \
    apt install --no-install-recommends -y python3.8 python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir torch torchvision torchaudio semilearn --index-url https://download.pytorch.org/whl/cu118

