
How to build and run:

- The docker image we build will use thecanadianroot/opencv-cuda:ubuntu20.04-cuda11.3.1-opencv4.5.2 as a base image.
- This is because it has installed CV2 with CUDA-ENABLED so we can use cv2.dnn modules.
- This way we can load the Darknet, YOLOv4 models using cv2, thus avoiding using subprocess with darknet.exe which is slow.



 docker build -t fish_elevated --file /mnt/c/Users/Woo-Jin/workspace/blue_altex_counting_detection/dockerfiles/Dockerfile.txt /mnt/c/Users/Woo-Jin/workspace/blue_altex_counting_detection/dockerfiles/



 docker run -it --rm --gpus=all -v /mnt/c/Users/Woo-Jin/workspace/blue_altex_counting_detection/:/workspace fish_elevated


IMPORTANT: gpus==0 DOES NOT WORK!! Must be gpus==all!