FROM continuumio/miniconda3:4.7.10
WORKDIR /train_model
COPY . /train_model
SHELL ["/bin/bash", "-c"]
RUN conda env create -f environment.yaml
RUN echo "source activate sa_ea_train_model" > ~/.bashrc
ENV PATH /opt/conda/envs/sa_ea_train_model/bin:$PATH
RUN python train_model.py
