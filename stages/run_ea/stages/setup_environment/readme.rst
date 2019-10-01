Creates a Docker image, which has the environment required to run
the EA.

In addtion, `run_ea.def` can be used to create a Singularity
image, referred to as `run_ea.sif` in this documentation, from the
Docker image. This requires the Docker image to be
saved into `/home/lukas/temp/sa_ea_run_setup_environment.tar`, which
means that `run_ea.def` may need to be edited and this path changed
for different users.

The Singularity image can be used to run the EA with::

$ singularity exec /path/to/run_ea.sif /path/to/run_ea.bash path/to/ea/input/file.py

where `run_ea.bash` is the file located in
`sa_ea/stages/run_ea/stages/run_ea/run_ea.bash`.
