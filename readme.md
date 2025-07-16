# Recommender System Research

## project setup:

### resote existing environment:

`conda env create -f environment.yml`
`conda create --name new_env_name --file requirements.txt`

for cuda enabled environment install the cuda and cuda enabled pyg lib

### update environment:

python version: `python=3.12`

```bash
conda env export > environment.yml
conda list --export > requirements.txt
```

### Git hooks setup

To automatically update environment files before committing:

#### Option 1: Manual git hook (already set up)

The pre-commit hook is already installed and will automatically update environment files before each commit.

#### Option 2: Using pre-commit framework

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run manually (optional)
pre-commit run --all-files
```
