# Shokunin July

Data is here: https://www.kaggle.com/c/synthetic-image-classification/data

```sh
kaggle competitions download -c synthetic-image-classification
```

## Environment 

```sh
brew install pyenv xz
pyenv init
# copy the output of above into .zhsrc (or whatever shell you use)

pyenv install 3.7.4
pyenv local
```

### Install dependencies

```sh
pip install -r requirements.txt
```

### Review the data exploration journey interactively

There is a notebook `shokunin-july-thoughtworks.ipynb`, you can launch `jupyter` locally to have a deep look of how the exploration was.

```sh
jupyter notebook
```

