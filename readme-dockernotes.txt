1. Build docker- (starts with nvidia cuda image, but not necessary with gemini)
docker build --no-cache --progress=plain -t rag-gpu-jupyter .

2. Run docker
docker run --gpus all -it -p 8888:8888 -v "$(pwd):/notebooks" rag-gpu-jupyter
After run starts, a link for jupyter notebook is created 

this kept crashing, so try after building container to install directly in docker container
# pip3 install --no-cache-dir flash-attn  --no-build-isolation --break-system-packages


