
![GitHub logo](./docs/full_logo.svg)

# Peytonites

This is the epository for the Princeton Astrophysical team at  NVIDIA + Princeton Open Hackathon 

## Welcome 

Welcome to the official GitHub repository for Team Peytonites! We are a passionate and dynamic team interested in GPU computing. Our team is comprised of talented individuals who specialize in various sub-fields. 

## Navigating

### [Peytonites Documentation](https://peytonites2024.readthedocs.io/en/latest/index.html)

Please refer to the documentation in the link above for more information. This repository contains the "peytonites" Python package, which is designed to assist in generating initial conditions, writing and reading from files, and providing some visualization tools. The purpose of this is to allow us to concentrate on the n-body aspect of the project.

The "hackathon" folder contains a serialized version of the n-body code that we hope to convert to GPU code. Please make sure to work in that directory. Also, remember not to push your simulations to GitHub. To help avoid mistakes, always include 'simout' somewhere in the output directory name

## Setup

To set up on the cluster:

Optional: if you like the gh tool, `wget https://github.com/cli/cli/releases/download/v2.50.0/gh_2.50.0_linux_amd64.tar.gz` and extract it to `~/.local`.

Get Hatch, then Python:

```bash
wget https://github.com/pypa/hatch/releases/latest/download/hatch-x86_64-unknown-linux-gnu.tar.gz
tar -xzf hatch-x86_64-unknown-linux-gnu.tar.gz -C ~/.local/bin
HATCH_PYTHON_VARIANT_LINUX=v2 hatch python install 3.12
```

> [!WARNING]
> Stellar is a heterogeneous cluster. If you don't specify
> `HATCH_PYTHON_VARIANT_LINUX=v2`, hatch will download the most optimized
> Python for the head node, which won't work on the workers.

Log out or re-source your config, however you want to get Python on the path.

Make sure CUDA is available:

```bash
ml cudatoolkit/12.4
```

That last line is optional, but keeps you from having to set it up later. Now you can use `hatch run cuda12:python` to start up a Python with everything present, including cupy!

If you want to build and serve the docs (locally, for example, not on the cluster), you can use `hatch run docs:serve`.

For Jupyter lab (such as locally), use:

```bash
hatch run jupyter:lab
```

## Submitting jobs

Make sure the environment is updated on the head node first:

```
hatch env create cuda12
```

(Or any `hatch run cuda12:...` command does this.)

To submit:

```bash
sbatch submit.sbatch
```

from the hackathon folder.
