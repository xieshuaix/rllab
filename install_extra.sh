#!/usr/bin/env bash
conda init && conda activate rllab3
cd 3rdparty
cd Theano && git checkout adfe319ce6b781083d8dc3200fb4481b00853791 && cd ..
cd Lasagne && git checkout 484866cf8b38d878e92d521be445968531646bb8 && cd ..
cd plotly.py && git checkout 2594076e29584ede2d09f2aa40a8a195b3f3fc66 && cd ..
cd gym && git checkout v0.7.4 && cd ..
pip install mujoco-py-v0.5.7
pip install Theano
pip install Lasagne
pip install plotly.py
pip install gym
pip install prettytensor
# install tensowflow_gpu==1.13.1 for CUDA 10
pip install tensorflow_gpu==1.13.1
# install tensorflow_gpu==1.12.0 for CUDA 9
#pip install tensorflow_gpu==1.12.0
# original instruction: install tensorflow 1.0.1
#pip install tensorflow_gpu-1.0.1-cp35-cp35m-linux_x86_64.whl
cd ..
# jupyter extensions
conda install -c conda-forge jupyter_contrib_nbextensions
pip install --upgrade nbdime
pip install papermill
