FROM sa_ea_run_ea_setup_environment
WORKDIR /run_ea
COPY . /run_ea
SHELL ["/bin/bash", "-c"]
RUN source activate sa_ea_run_ea; \
    PYTHONPATH=/setup_environment/stk/src python -m stk.ea input1.py  &> \
    input1.output
