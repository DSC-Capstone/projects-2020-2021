#!/usr/bin/env bash

NVIDIA_PATH=/usr/local/nvidia/bin

grep -qF "${NVIDIA_PATH}" ~/.bashrc || echo "export PATH=$PATH:${NVIDIA_PATH}" >> ~/.bashrc

jupyter notebook "$@"
