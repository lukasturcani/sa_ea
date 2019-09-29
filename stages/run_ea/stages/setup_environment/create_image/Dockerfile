FROM train_model AS TRAIN_MODEL
FROM alpine/git:1.0.7 AS STK

WORKDIR /opt
COPY ./gitcredentials /

RUN git config --global credential.helper 'store --file /gitcredentials'; \
    git clone https://github.com/lukasturcani/stk_ea /opt/stk; \
    cd /opt/stk; \
    git checkout f365c07a28f35da5217c67b612398ad33bf0ccdc

FROM continuumio/miniconda3:4.7.10

WORKDIR /setup_environment
ADD . /setup_environment
COPY --from=TRAIN_MODEL /train_model/sa_model.pkl /setup_environment
COPY --from=STK /opt/stk /setup_environment/stk

SHELL ["/bin/bash", "-c"]
RUN conda env create -f environment.yaml
# RUN
