
![GitHub logo](./docs/full_logo.svg)

# Peytonites

This is the epository for the Princeton Astrophysical team at  NVIDIA + Princeton Open Hackathon 

# How to run this branch
```shell
git clone https://github.com/robelgeda/peytonites2024.git
```
```shell
cd peytonites2024
```
```shell
conda env create --file=environment.yml
```
```shell
conda activate peytonites
```
```shell
python install -e
```
Now you can run things just normally or with the included job.sh that is in the hackathon directory. Can test by
```shell
cd hackathon
```
```shell
sbatch job.sh
```


## Welcome 

Welcome to the official GitHub repository for Team Peytonites! We are a passionate and dynamic team interested in GPU computing. Our team is comprised of talented individuals who specialize in various sub-fields. 

## Navigating

### [Peytonites Documentation](https://peytonites2024.readthedocs.io/en/latest/index.html)

Please refer to the documentation in the link above for more information. This repository contains the "peytonites" Python package, which is designed to assist in generating initial conditions, writing and reading from files, and providing some visualization tools. The purpose of this is to allow us to concentrate on the n-body aspect of the project.

The "hackathon" folder contains a serialized version of the n-body code that we hope to convert to GPU code. Please make sure to work in that directory. Also, remember not to push your simulations to GitHub. To help avoid mistakes, always include 'simout' somewhere in the output directory name
