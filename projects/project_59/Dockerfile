FROM ucsdets/scipy-ml-notebook

USER root


# from amfraenkel/android-malware-project
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y default-jre && \
    apt-get install -y default-jdk && \
    apt-get install -y software-properties-common && \
    apt-get install -y intel-mkl-full

ENV APK_SCRIPT https://raw.githubusercontent.com/iBotPeaches/Apktool/master/scripts/linux/apktool
ENV APK_JAR https://bitbucket.org/iBotPeaches/apktool/downloads/apktool_2.4.1.jar

RUN mkdir -p /usr/local/bin

RUN P=/tmp/$(basename $APK_SCRIPT) && \
    wget -q -O $P $APK_SCRIPT && \
    chmod +x $P && \
    mv $P /usr/local/bin

RUN P=/tmp/$(basename $APK_JAR) && \
    wget -q -O $P $APK_JAR && \
    chmod +x $P && \
    mv $P /usr/local/bin/apktool.jar

# Tensorflow dependencies to avoid warnings. See: https://www.tensorflow.org/install/gpu

# Add NVIDIA package repositories
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
# RUN apt-get update -y

# RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

# RUN apt install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
# RUN apt-get update -y

# # Install NVIDIA driver
# RUN apt-get install -y --no-install-recommends nvidia-driver-450 
# # Reboot. Check that GPUs are visible using the command: nvidia-smi

# RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
# RUN apt install -y ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
# RUN apt-get update -y

# # Install development and runtime libraries (~4GB)
# RUN apt-get install -y --no-install-recommends --allow-downgrades \
#     cuda-11-0 \
#     libcudnn8=8.0.4.30-1+cuda11.0  \
#     libcudnn8-dev=8.0.4.30-1+cuda11.0 


# # Install TensorRT. Requires that libcudnn8 is installed above.
# RUN apt-get install -y --no-install-recommends --allow-downgrades \
#     libnvinfer7=7.1.3-1+cuda11.0 \
#     libnvinfer-dev=7.1.3-1+cuda11.0 \
#     libnvinfer-plugin7=7.1.3-1+cuda11.0

# Additional py packages
RUN pip install stellargraph p_tqdm jekyllnb adversarial-robustness-toolbox sparse-dot-mkl cupy-cuda111 
     
RUN export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH

    