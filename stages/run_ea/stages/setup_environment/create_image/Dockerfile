FROM train_model AS TRAIN_MODEL
FROM alpine/git:1.0.7 AS STK

WORKDIR /opt
COPY ./gitcredentials /
git config --global credential.helper 'store --file /gitcredentials'
RUN git clone https://github.com/lukasturcani/stk_ea stk; \
    cd /opt/stk; \
    git checkout d7a1e31008cf720e165c0e8045331156241d5bfd

FROM continuumio/miniconda3:4.7.10

WORKDIR /setup_environment
ADD . /setup_environment
COPY --from=TRAIN_MODEL /train_model/train_model.pkl /setup_environment
COPY --from=STK /opt/stk /setup_environment

SHELL ["/bin/bash", "-c"]
RUN conda env create -f environment.yaml
RUN 
