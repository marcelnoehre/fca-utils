# FCA Utilities
Utility functions for research in the field of Formal Concept Analysis (FCA)

## Installation Instructions

### Kissat SAT Solver

#### 1. Download the latest release from [Kissat GitHub](https://github.com/arminbiere/kissat/).
#### 2. Rename the downloaded binary for convenience

```bash
mv kissat-<$version>-linux-amd64 kissat
```

#### 3. Move the binary to a local user directory and make it executable

```bash
mkdir -p ~/bin
mv kissat ~/bin/
chmod +x ~/bin/kissat
```

#### 4. Add the directory to the `PATH` environment variable

```bash
export PATH="$HOME/bin:$PATH"
```

#### 5. Verify the installation

```bash
source ~/.bashrc
which kissat
kissat --help
```

---

### Miniforge (Python Environment Manager)

#### 1. Download the latest Miniforge installer from [Miniforge Downloads](https://conda-forge.org/download/).
#### 2. Make the installer executable and run it

```bash
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
```

---

### Conda Environment Setup

#### 1. Create a new conda environment `fca-utils` with SageMath and Python 3.11

```bash
mamba create -n fca-utils sage python=3.11
```

#### 2. Activate the environment

```bash
conda activate fca-utils
```

#### 3. Install required Python packages

```bash
pip install -r requirements.txt
```

#### 4. Include the local bin directory in the conda environment `PATH`

```bash
export PATH="$HOME/bin:$PATH"
```

---

### Running Python Scripts

Use SageMath's Python interpreter to run scripts:

```bash
sage -python <file>.py
```

---
