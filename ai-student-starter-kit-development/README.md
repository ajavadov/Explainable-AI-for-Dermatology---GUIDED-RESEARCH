# pl-starter-kit

Small codebase to be used as a test barebone by new students on our compute server.

## Getting started (Conda based approach)

Install miniconda in your home.

Set the `PATH` to point to the miniconda installation folder.
If it's installed in your home dir then it should look like this:
```
export PATH="~/miniconda3/bin:$PATH"
```

```
conda create --name <env_name> python=3.8 -y
conda activate <env_name>
```

## Package installation

```
source activate <env_name>
python3 -m pip install -r requirements.txt
```

## Running with SLURM

Replace the `<env_name>` in the slurm_script.sh file with the correct conda environment name.
Choose a name for the current job: `<job_name>`

Then call:
```
sbatch -w cmpsrv01.ciip.in.tum.de slurm_script.sh
```

## Getting started (Poetry based approach)

Please provide steps here

