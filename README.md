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

### Code Structure

- `main.py` for training the model
- `predicate.py` for using the model for predication and generate output

We're employing [`EfficientNetB3` model](https://arxiv.org/abs/1905.11946) as the first half of the classification, and the simple `full-connected-network` to classify different labels.