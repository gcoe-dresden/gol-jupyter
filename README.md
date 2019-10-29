# gol-jupyter
Game of Life - Jupyter Notebook

## Start Jupyter Notebook

To start the jupyter server for interactive CUDA programming follow these commands:
```bash
# connect to taurus head node
$ ssh taurus.hrsk.tu-dresden.de
# allocation of an interactive K20X GPU using reservation and project IDs for 90min
$ srun -A p_lv_gpu1920 --reservation=p_lv_gpu1920_319 -p gpu1-interactive --gres=gpu:1 --mem=4000 -t 90 --pty bash -i –l
# check you got a taurusi20** compute node and keep the number in mind:
$ hostname
# clone repository & cd into it
$ git clone https://github.com/gcoe-dresden/gol-jupyter.git gol-jupyter
$ cd gol-jupyter
# run singularity container on the compute node, COPY the link and keep the port number in mind
$ singularity run --nv /projects/p_lv_gpu1920/lab03/GOL_TU_ex.sif
# the following port number 8888 must be replaced accordingly
# open another shell and create a tunnel to the head and to the compute node
$ ssh taurus.hrsk.tu-dresden.de -L 8888:localhost:8888
# on the headnode you create another tunnel to the compute node from above, eg. taurusi2014
$ ssh taurusi20**.taurus.hrsk.tu-dresden.de -L 8888:localhost:8888
# Open the link from above (you also can click on the link in the output of the jupyter server)
# "Jupyter Files" allows you to upload and to open the provided notebook!
```
