# ProtonDB Data Project
## Getting Started

This project uses Jupyter Notebook to create a trainset for ProtonDB data. To get started, follow these steps:

### Step 1: Extract Project Files
Extract the project files from the GitHub repository into the /data directory:

```bash
mkdir data
cd data
git clone https://github.com/bdefore/protondb-data
```

### Step 2: Register Jupyter Kernel
Register the Jupyter kernel in your virtual environment (venv):

```bash
jupyter kernelspec install /path/to/your/kernel/spec/file.json --user --name caniplayonlinux
```
Replace /path/to/your/kernel/spec/file.json with the actual path to your kernel spec file.

### Step 3: Launch Jupyter Server
Launch the Jupyter server:

```bash
jupyter lab
```

### Step 4: Register Remote Kernel in IDE

Copy the token from the command line output and register the remote kernel in your IDE.

### Step 5: Run Notebooks
Run the caniplayonlinux.ipynb notebook to create the trainset in the /data directory.

### Prerequisites
* Jupyter Notebook installed
* Virtual environment (venv) set up
* IDE with Jupyter Notebook support

### Troubleshooting
* Make sure to replace /path/to/your/kernel/spec/file.json with the actual path to your kernel spec file.
* If you encounter issues with the Jupyter server, try restarting it or checking the logs for errors.