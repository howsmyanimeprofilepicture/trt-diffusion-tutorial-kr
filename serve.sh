docker run -ti --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 3691:8000 \
    -v ./repo:/repo \
    --name triton-server \
    triton \
    tritonserver --model-repository=/repo --exit-on-error=false 
