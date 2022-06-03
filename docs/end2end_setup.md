# Setup Soft-teacher repo with CBNv2 

## Clone repo
```
git clone https://github.com/thangnx183/SoftTeacher.git
cd SoftTeacher
git checkout feat/MCAI-194-setup-mask-model
```

## Setup docker
- build docker image
    ```
    docker build -t steacher docker/
    ```

- build docker container
    ```
    nvidia-docker run --name st --gpus all --shm-size=64g -p 7000:7000 -p 7001:7001 -it -v /mnt/ssd1/thang/SoftTeacher/:/SoftTeacher -v /mnt/ssd1/thang/coco/datasets/:/SoftTeacher/data steacher

    #/mnt/ssd1/thang/SoftTeacher/ : path of code folder
    #/mnt/ssd1/thang/coco/datasets/ : path to coco datasets format
    ```

## Setup thirdparty (CBNv2) and final install
```
make install
```

### More libs
- mmcv
    ```
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    git checkout db097bd1e97fc446a7551c715970611d2fcc848d
    MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_86' pip install -e .
    cd ..
    ```

- apex
    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
    python setup.py install --cuda_ext --cpp_ext
    cd ..
    ```