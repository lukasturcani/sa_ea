FROM train_model AS TRAIN_MODEL
FROM alpine/git:1.0.7 AS INSTALL

WORKDIR /opt
COPY ./install_stage /opt

RUN git config --global credential.helper 'store --file /opt/gitcredentials'; \
    git clone https://github.com/lukasturcani/stk_ea /opt/stk; \
    cd /opt/stk; \
    git checkout f365c07a28f35da5217c67b612398ad33bf0ccdc

FROM continuumio/miniconda3:4.7.10

WORKDIR /setup_environment
COPY ./Dockerfile ./create_image.bash ./setup_stage/setup_environment /setup_environment/
COPY --from=TRAIN_MODEL /train_model/sa_model.pkl /setup_environment
COPY --from=INSTALL /opt/stk /setup_environment/stk
COPY ./setup_stage/schrodinger2018-1 /opt/schrodinger2018-1
COPY ./setup_stage/xtb_190418 /opt/xtb_190418

SHELL ["/bin/bash", "-c"]
RUN conda env create -f environment.yaml
