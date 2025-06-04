- - - <h2 align="center">
      <a href="https://neurips.cc/virtual/2024/poster/94253" target="_blank">LinNet: Linear Network for Efficient Point Cloud Representation Learning</a>
      </h2>


      <h3 align="center">
      NeurIPS 2024
      </h3>

      


      ## 🔧  Installation
    
      Please use the following command for installation.
    
      ```bash
      # It is recommended to create a new environment
      conda create -n linnet python==3.8 -y
      conda activate linnet
      
      conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
      
      conda install ninja -y
      pip install h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm
      
      conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
      pip install torch-geometric
      
      # spconv (SparseUNet)
      # refer https://github.com/traveller59/spconv
      pip install spconv-cu113
      
      # PTv1 & PTv2 or precise eval
      cd libs/pointops
      # usual
      python setup.py install
      cd ../..
      
      # Hash Query
      cd linnetops
      python setup.py install
      cd ..
      ```
    
      The code has been tested on Ubuntu 20.04 with GCC 9.4.0, Python 3.8, PyTorch 1.12.1, and CUDA 11.6, using an NVIDIA 4090D GPU.


      ## 🚅 Quick Start
      We provide training scripts for nuScenes with the following commands:
    
      ### Training on nuScenes
    
      For dataset preparation, please refer to the instructions in the [Pontcept](https://github.com/Pointcept/Pointcept/blob/e384a8a4cd9f24aeb084740add03a9820c5cb2e8/README.md?plain=1#L408)
    
      Before starting training, organize the dataset in the following structure:
    
      ```bash
      nuscene
      |── raw
          │── samples
          │── sweeps
          │── lidarseg
          ...
          │── v1.0-trainval
          │── v1.0-test
      |── info
      ```
      Traing LinNet with 4 GPUs:
      ```bash
      CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d nuscenes -c semseg-linnet -n semseg-linnet
      ```


      ## ⛳ Testing
      To test a pre-trained models on nuScenes, use the following commands:
      ```bash
      sh scripts/test.sh -p python -d nuscenes -n semseg-linnet -w model_best -g 1
      ```
    
      ## 📚 Citation
      If you find this work useful, please consider citing:
      ```bibtex
      @inproceedings{deng2024linnet,
        title={Linnet: Linear network for efficient point cloud representation learning},
        author={Deng, Hao and Jing, Kunlei and Cheng, Shengmei and Liu, Cheng and Ru, Jiawei and Bo, Jiang and Wang, Lin},
        journal={Advances in Neural Information Processing Systems},
        volume={37},
        pages={43189--43209},
        year={2024}
      }
      ```
    
      ## 🙏 Acknowledgements
      Our code is heavily brought from
    
      - [Pointcept](https://github.com/Pointcept/Pointcept)

