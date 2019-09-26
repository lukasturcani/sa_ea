FROM continuumio/miniconda3:4.7.10
WORKDIR /train_model
COPY . /train_model
SHELL ["/bin/bash", "-c"]
RUN conda env create -f environment.yaml
RUN source activate sa_ea_train_model; python train_model.py
