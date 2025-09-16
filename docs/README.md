# Documentation

## Install requirements

```bash
cd docs
pip install -r requirements.txt
```

## Build the documentation

```bash
cd docs
sphinx-apidoc -f -o api ../edugrid
make html
```

## Clean the documentation

```bash
cd docs
make clean
```
