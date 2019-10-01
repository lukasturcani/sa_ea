Creates a Docker image, which has the environment required to run
the EA.

In addtion, `run_ea.def` can be used to create a Singularity
image from the Docker image. This requires the Docker image to be
saved into `/home/lukas/temp/sa_ea_run_setup_environment.tar`, which
means that `run_ea.def` may need to be edited for different users.

The Singularity image can be used to run the EA with::

$ singularity exec run_ea.sif run_ea.bash path/to/ea/input/file.py

where `run_ea.bash` is the file located in this directory.
