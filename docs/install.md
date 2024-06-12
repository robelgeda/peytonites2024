# Setup and Installation

## Cluster Setup

To set up on the cluster do the following

Get Hatch, then Python:

1. Download Hatch:
```bash
wget https://github.com/pypa/hatch/releases/latest/download/hatch-x86_64-unknown-linux-gnu.tar.gz
```
2. Make a `~/.local/bin` folder incase you never set it up before, if its there nothing happens:
```bash
mkdir -p ~/.local/bin
```

3. Unpack Hatch into your `~/.local/bin`:
```bash
tar -xzf hatch-x86_64-unknown-linux-gnu.tar.gz -C ~/.local/bin
```

4. Install Python using Hatch
```bash
HATCH_PYTHON_VARIANT_LINUX=v2 hatch python install 3.12
```

```{WARNING}
Stellar is a heterogeneous cluster. If you don't specify
`HATCH_PYTHON_VARIANT_LINUX=v2`, hatch will download the most optimized
Python for the head node, which won't work on the workers.
```

5. Log out or re-source your config, however you want to get Python on the path.

## CUDA Setup

1. Make sure CUDA is available:

```bash
ml cudatoolkit/12.4
```

That last line is optional, but keeps you from having to set it up later. 

2. If you haven't already cloned this repo, please do so and cd into it as follows:

```bash
git clone https://github.com/robelgeda/peytonites2024.git
```

```bash
cd peytonites2024
```

3. Now you can use the command below to start up a Python terminal with everything present, including cupy!

```bash
hatch run cuda12:python
```

You can run `import cupy` to check if everything worked:

```bash
Python 3.12.3 (main, Apr 15 2024, 17:48:30) [Clang 17.0.6 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cupy
```

**Build Docs Locally (optional):**

If you want to build and serve the docs (locally, for example, not on the cluster), you can use `hatch run docs:serve`.

**Jupyter Lab (optional):**

For Jupyter lab (such as locally), use:

```bash
hatch run jupyter:lab
```

If you want to use Jupyter on the cluster, run this command:

```bash
hatch run cuda12:ipykernel
```

This will install a file in your user directory that allows Jupyter to find the
kernel, which will be named "cupy12". Same thing ccan be done for the "jax"
environment, with a matching name.

## Submitting jobs

```{TIP}
You should put all your code in the `peytonites2024/hackathon/main.py` file.
```

1. The job submission material is in the `peytonites2024/hackathon` subfolder for the repository. Before you go to the next step, please ensure you are in that dir:

```bash
pwd 
```
or 
```bash
cd hackathon
```

2. Make sure the environment is updated on the head node first:

```bash
hatch env create cuda12
```

(Or any `hatch run cuda12:...` command does this, like `hatch run cuda12:echo Synced`)

3. To submit:

```bash
sbatch submit.sbatch
```

from the hackathon folder.


## gh Tool

Optional: if you like the gh tool, you can install it like this:

```bash
wget https://github.com/cli/cli/releases/download/v2.50.0/gh_2.50.0_linux_amd64.tar.gz
```
```bash
mkdir ~/.local
```
```bash
tar -xzf gh_2.50.0_linux_amd64.tar.gz --strip-components=1 -C ~/.local
```

## Conda

If you want to use conda, you can:

```bash
module load anaconda3/2023.3 cudatoolkit/12.4
source /usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh
conda env create -n peytonites python==3.12 pip
conda activate peytonites
pip install -e .[cuda12,jax]
```

Use the following lines in your batch jobs instead:

```bash
module load anaconda3/2023.3 cudatoolkit/12.4
source /usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh
conda activate peytonites
```
