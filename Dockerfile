FROM kundajelab/cuda-anaconda-base

# Please make sure to run these before docker build
#python -m scripts.download_cvc --model_type CVC
#python -m scripts.download_cvc --model_type scCVC
#python -m scripts.download_cvc_training_data --model_type CVC
#python -m scripts.download_cvc_training_data --model_type scCVC

# Docker build command
#docker build --progress=plain   -t cvc_env .
#Docker run command:
#docker run --gpus all -it cvc_env

RUN conda update -n base -c defaults conda -y && conda clean -afy
WORKDIR /app
COPY environment.yml /app
RUN conda env create -f environment.yml

# Copy entire project after environment setup to optimize cache usage
COPY . /app

# Activate the conda environment and run tests
RUN conda run -n cvc python -m tests.test_create_embeddings
