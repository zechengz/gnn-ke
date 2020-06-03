## Link Prediction with Graph Neural Networks and Knowledge Extraction

### Installation

```shell
pip install -r requirements.txt
```

Maybe use `pip3` instead of `pip`.

To manuall install the PyTorch Geometric, please refer to the [installation page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### Python Version

> Python version >= python3.6

### Data

To get the data, please download the data and place them to the `data` folder:

```shell
wget https://www.dropbox.com/s/22czhgi12y4j4us/bern.pkl
wget https://www.dropbox.com/s/chrj503ru9x406f/reference_tensor.gpickle
```

### Example

```shell
cd src
python link_pred.py --device=cpu
```
