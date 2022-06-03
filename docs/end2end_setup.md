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
