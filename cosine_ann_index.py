from typing import Any
import fdb
import fdb.tuple
import numpy as np
from collections import deque

class CosineANNIndex (object):
    def _projection_vector(self, dims: int, seq: int):
        '''Return the `seq`-th projection vector of length D, 
        where `seq` is an integer between 0 and N-1'''
        rng = np.random.RandomState(seed=seq)
        vector = rng.normal(0, 1, dims)
        return vector

    def _get_projection_vectors(self, N: int, dims: int):
        '''Return a list of N projection vectors of length D'''
        return [self._projection_vector(dims, i) for i in range(N)]

    def _hash_function(self, vector: np.array):
        '''Return the hash of the vector using the projection vectors'''
        return sum((1 << i) for i, pv in enumerate(self.projection_vectors) if np.dot(vector, pv) >= 0)
    
    def _adjacent_hashes(self, hash: int):
        return [hash ^ (1 << i) for i in range(1, self.N)]
    
    def __init__(self, dims: int, N: int = 8, prefix: bytes = b'_cosann'):
        self.dims = dims
        self.N = N
        self.prefix = prefix
        self.projection_vectors = self._get_projection_vectors(N, dims)
    
    @fdb.transactional
    def add(self, tr: fdb.Transaction, id: Any, vec: np.array):
        hash = self._hash_function(vec)
        tr[fdb.tuple.pack((hash, id), self.prefix)] = fdb.tuple.pack(tuple(vec))
    
    @fdb.transactional
    def remove(self, tr: fdb.Transaction, id: Any):
        hash = self._hash_function(vec)
        del tr[fdb.tuple.pack((hash, id), self.prefix)]
    
    @fdb.transactional
    def query(self, tr: fdb.Transaction, id: Any, vec: np.array, desired_neighbor_count: int, recall_boost_factor=0):
        vec_hash = self._hash_function(vec)
        startswith = lambda hash: tr.get_range_startswith(fdb.tuple.pack((hash,), self.prefix))

        # BFS hash regions to find nearest neighbors
        q = deque([(vec_hash, 0)])
        neighbors = []
        queued_hashes = {}
        max_depth = 999
        while len(q) > 0:
            hash, depth = q.popleft()
            if depth > max_depth:
                break
            for k, v in startswith(hash):
                _, neighbor_id = fdb.tuple.unpack(k, len(self.prefix))
                if neighbor_id == id:
                    continue
                neighbor = np.array(fdb.tuple.unpack(v))
                cos_sim = np.dot(vec, neighbor) / np.linalg.norm(vec) / np.linalg.norm(neighbor)
                neighbors.append((neighbor_id, cos_sim))
            if len(neighbors) >= desired_neighbor_count:
                max_depth = depth + recall_boost_factor
            for adjacent_hash in self._adjacent_hashes(hash):
                if adjacent_hash not in queued_hashes:
                    queued_hashes[adjacent_hash] = True
                    q.append((adjacent_hash, depth + 1))
        
        # Sort neighbors by cosine similarity descending
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        return neighbors[:desired_neighbor_count]