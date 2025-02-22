# FDB ANN Index

This repository contains an implementation of an Approximate Nearest Neighbors (ANN) index using locality sensitive hashing as a layer for FoundationDB. It currently supports only cosine similarity as the similarity metric.

## Installation

Copy the `cosine_ann_index.py` file into your project. Ensure that all dependencies are met in the `requirements.txt` file. (I don't plan to publish this on PyPI, sorry.)

## Usage

### Initialization

```python
from cosine_ann_index import CosineANNIndex

# Create an index for a vector space with specified dimensions
index = CosineANNIndex(dims=128, N=8)
```

### Adding a Vector

You can use `index.add` to add a vector to the index. The `id` parameter is a unique identifier for the vector, which will be returned later when querying the index, and the `vec` parameter is the vector itself. The first parameter is a transaction to use, or the database itself if a new transaction should be created. This is true for all index methods.
```python
index.add(tr, id=1, vec=np.array([0.1, 0.2, ..., 0.128]))
```

### Removing a Vector

```python
index.remove(tr, 1)
```

### Querying Neighbors

You can use `index.query` to query the index for the nearest neighbors to a given vector. The `vec` parameter is the vector to query. The `id` parameter is the ID of the vector, which is used for ensuring that the same vector is not returned as a neighbor - this can be set to `None` if you don't want to exclude any specific vector. The `desired_neighbor_count` parameter is the number of neighbors to return. The method returns a list of tuples, where each tuple contains the ID of the nearest neighbor and the similarity score, in descending order of similarity.

```python
neighbors = index.query(tr, 1, vec=np.array([0.1, 0.2, ..., 0.128]), desired_neighbor_count=5)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.