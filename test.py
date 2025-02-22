import fdb
fdb.api_version(730)

from cosine_ann_index import CosineANNIndex
import numpy as np

db = fdb.open("data/fdb.cluster")
# Clear test database
db.clear_range(b'\x00', b'\xff')

N = 10
dims = 40
vec_count = 400
index = CosineANNIndex(dims, N)
vecs = np.random.normal(0, 1, (vec_count, dims))

@fdb.transactional
def construct_index(tr: fdb.Transaction):
    for i, vec in enumerate(vecs):
        index.add(tr, i, vec)
construct_index(db)

def get_exact_neighbors(vec_id, vec, neighbor_count):
    cos = lambda x, y: np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)
    all_neighbors = [(neighbor, cos(vec, neighbor), id) for id, neighbor in enumerate(vecs) if id != vec_id]
    all_neighbors.sort(key=lambda x: x[1], reverse=True)
    return [(id, cos_sim) for _neighbor, cos_sim, id in all_neighbors[:neighbor_count]]

neighbor_count = 5
approximate_neighbor_ids = index.query(db, 0, vecs[0], neighbor_count)
exact_neighbor_ids = get_exact_neighbors(0, vecs[0], neighbor_count)

def hash_dist(h1, h2):
    xor = h1 ^ h2
    popcount = 0
    while xor > 0:
        popcount += xor & 1
        xor >>= 1
    return popcount

vec0_hash = index._hash_function(vecs[0])

print("Approximate:")
for x in approximate_neighbor_ids:
    print(x, end=' ')
    print("hash_dist =", hash_dist(index._hash_function(vecs[x[0]]), vec0_hash))

print("Exact:")
for x in exact_neighbor_ids:
    print(x, end=' ')
    print("hash_dist =", hash_dist(index._hash_function(vecs[x[0]]), vec0_hash))