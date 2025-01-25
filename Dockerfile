FROM kundajelab/cuda-anaconda-base

# Please make sure to run these before docker build
#python -m scripts.download_cvc --model_type CVC
#python -m scripts.download_cvc --model_type scCVC
#python -m scripts.download_cvc_training_data --model_type CVC
#python -m scripts.download_cvc_training_data --model_type scCVC

# Docker build command
#docker build --progress=plain   -t cvc_env .
#Docker run command:
# docker run --gpus all -it \
#    -v ./data/:/app/data/  cvc_env \
#    --input_path /app/data/custom_dataset/test_tcrs.csv  \
#    --output_path /app/data/custom_dataset/embeddings.csv

RUN conda update -n base -c defaults conda -y && conda clean -afy
WORKDIR /app
COPY environment.yml /app
RUN conda env create -f environment.yml

# Copy entire project after environment setup to optimize cache usage
COPY . /app

# Activate the conda environment
RUN echo "source activate cvc" > ~/.bashrc
ENV PATH /opt/conda/envs/cvc/bin:$PATH

# Set the default command to execute the script
ENTRYPOINT ["conda", "run", "-n", "cvc", "python", "-m", "scripts.create_embeddings"]
