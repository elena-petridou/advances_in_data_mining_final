"""
File containing the LSH class for the execution of the algorithm and helper functions.
"""
from __future__ import annotations
from itertools import combinations
from xxhash import xxh32
import numpy as np
from scipy.sparse import csr_matrix
import sympy as sp

class LSH:
    """
    Class for processing, banding and hashing 
    the data for the LSH algorithm.
    """
    def __init__(self, data: csr_matrix):
        self.data_ = data
        self.users_, self.movies_ = data.get_shape()
        self.candidate_pairs_ = set()
        self.thresh_ = 0.5

    def generate_hash_functions(self,
                                seed: int,
                                num_functions: int) -> np.ndarray:
        """
        Generates a set of hash functions with expression (ax + b) mod c. a and b are random 
        integers between 1 and max movie id, c is a prime number larger than max movie id.
        Args:
            seed (int): Seed for random number generator
            num_functions (int): Number of hash functions to generate
        Returns:
            dict[str, np.ndarray]: Dictionary containing the hash function parameters
        """
        # Create an array to hold the hash function parameters
        self.hash_functions_ = np.empty((num_functions, 3), dtype=np.int64)

        # Getting some useful variables
        data = self.data_
        movies = data.indices
        self.num_functions_ = num_functions

        # Getting max movie id for hash function generation
        # (we need this to define our c since it needs to be
        # a prime number larger than the length of the data)
        max_value = int(movies.max())
        c = find_next_prime(max_value)

        # Setting seed (taken from args)
        np.random.seed(seed)
        # Generating random a, b, c values for each hash function
        self.hash_functions_[:,0] = np.random.randint(1, c, size=num_functions)
        self.hash_functions_[:,1] = np.random.randint(1, c, size=num_functions)
        self.hash_functions_[:,2] = c
        return self.hash_functions_

    def minhashing(self) -> np.ndarray:
        """
        Applies MinHashing to all the data using vectorised operations 
        for speed using the hash function (a * movie_ids + b) mod c.

        Returns:
            np.ndarray: MinHash signatures for all users
        """
        # Create an array to hold the signatures
        self.signatures_ = np.full((self.users_, self.num_functions_), 0)
        # Setting the arrays of parameters
        a = self.hash_functions_[:,0].astype(np.int32).reshape(-1,1)
        b = self.hash_functions_[:,1].astype(np.int32).reshape(-1,1)
        c = self.hash_functions_[:,2].astype(np.int32).reshape(-1,1)
        movie_ids = np.arange(self.movies_).reshape(1, -1)
        # Computing the signatures for movies (returns array of shape (num_functions, num_movies))
        hash_matrix = (a*movie_ids + b)%c
        # For each user, find the movies they rated and compute the min hash values
        for user in range(self.users_):
            rated_movies = self.data_[user].indices
            # These lines resolve a seeming bug with the first entry in the signature matrix
            if rated_movies.size == 0:
                continue
            # Slice only the columns of the hash matrix which contain
            # movies the user has rated and take the minimum hash value for each hash function
            self.signatures_[user] = hash_matrix[:, rated_movies].min(axis=1)
        return self.signatures_

    def band_signatures(self, num_bands: int) -> np.ndarray:
        """
        Function implementing the banding of the signatures and finding candidate pairs
        Args:
            num_bands (int): Number of bands to create
        Returns:
            np.ndarray: Banded data array
        """
        self.num_bands_ = num_bands
        self.banded_data_ = np.empty((self.users_, num_bands), dtype=int)
        # Calculate number of columns per band (floor division in case of non-even division)
        cols_per_band = self.num_functions_ // num_bands
        for b in range(num_bands):
            start = b * cols_per_band
            if b == num_bands - 1: # last band takes the remaining rows
                end = self.num_functions_
            else:
                end = start + cols_per_band
            # Slice the columns (hash functions) for this band (shape: (num_users, cols_per_band))
            block = self.signatures_[:, start:end]
            # Hash each user using deterministic hashing (shape: (num_users, num_bands))
            self.banded_data_[:,b] = np.apply_along_axis(lambda row:
                xxh32(row.tobytes()).intdigest(), 1, block)
        return self.banded_data_

    def find_candidate_pairs(self, thresh) -> set[tuple[int, int]]:
        """
        Finds candidate pairs from the banded data.
        Args:
            thresh (int): Size limit for candidate pairs per bucket
        Returns:
            set[tuple[int, int]]: Set of candidate user pairs
        """
        # Initialize an empty set to hold candidate pairs
        candidate_pairs = set()
        # For each band, create buckets and find candidate pairs
        for b in range(self.num_bands_):
            # Empty bucket for the band
            buckets = {}
            # For each user, use the hash values as bucket keys and add user
            # as value to the corresponding key, so that users with
            # same hash value go into the same bucket
            for u in range(self.users_):
                hash_value = self.banded_data_[u, b]
                try:
                    # Limit the size of buckets to decrease the number of false positives
                    if len(buckets[hash_value]) < thresh:
                        buckets[hash_value].append(u)
                except KeyError:
                    buckets[hash_value] = []
                    buckets[hash_value].append(u)

            # Make pairs from users in the same bucket and add to candidate pairs set
            for bucket in buckets.values():
                # Limit the size of buckets we evaluate to reduce computational demands
                if len(bucket) > 1:
                    candidate_pairs.update(combinations(sorted(bucket), 2))
        self.candidate_pairs_ = candidate_pairs
        return candidate_pairs

    def compute_pseudo_jaccard(self) -> np.ndarray:
        """
        Computes pseudo-Jaccard similarities for candidate 
        pairs by making use of their MinHash signatures.

        Returns:
            np.ndarray: Array of user pairs with their pseudo-similarities
        """
        # Convert candidate pairs set to numpy array for vectorized operations
        pairs = np.array(list(self.candidate_pairs_))
        pairs_per_band = 10000
        num_bands = len(pairs)//pairs_per_band
        similarities = np.empty((len(pairs)), dtype=np.float32)
        for b in range(num_bands):
            start = b * pairs_per_band
            if b == num_bands - 1: # last band takes the remaining rows
                end = len(pairs)
            else:
                end = start + pairs_per_band
            # Slice the pairs for this band
            block = pairs[start:end]
            # Extract arrays of users from pairs
            u1 = block[:, 0]
            u2 = block[:, 1]
            # Extract arrays of signatures for each user in the pairs
            s1 = self.signatures_[u1]
            s2 = self.signatures_[u2]
            # Calculate pseudo-Jaccard similarities as the fraction of matching
            # minhashes (u1 intersection u2) / (num_functions)
            similarities[start:end] = np.mean(s1 == s2, axis=1)
            # Stack the user pairs with their similarities (already
            # with u1<u2, as stated in the assignment)
        similarities = np.column_stack((pairs[:,0], pairs[:,1], similarities))
        self.similarities_ = similarities[similarities[:, 2] > self.thresh_, :].astype(float)
        self.pairs_ = similarities[similarities[:, 2] > self.thresh_, :2]
        return self.similarities_

    def compute_real_jaccard(self) -> float:
        """
        Computes the real Jaccard similarities for the candidate pairs with
        similarity above the threshold and calculates the true positive rate for the LSH algorithm.

        Returns:
            float: True positive rate
        """
        pairs = self.pairs_
        similarities = np.empty((len(pairs)), dtype=np.float32)

        for i, (user1, user2) in enumerate(pairs):
            # Get the item indices for each user (non-zero entries)
            items1 = set(self.data_[int(user1)].indices)
            items2 = set(self.data_[int(user2)].indices)

            # Compute Jaccard similarity
            intersection_size = len(items1 & items2)
            union_size = len(items1 | items2)
            if union_size > 0:
                similarities[i] = intersection_size / union_size
            else:
                similarities[i] = 0.0
        proportion = float(np.sum(similarities > self.thresh_)/len(similarities))
        self.true_similarities_ = similarities
        self.true_pairs_ = pairs[similarities > self.thresh_]
        self.true_positive_rate_ = proportion
        return proportion

def find_next_prime(start:int) -> int:
    """
    Finds the next prime number greater than or equal to start.
    Args:
        start (int): Starting number to check for primality
    Returns:
        int: Next prime number greater than or equal to start
    """
    while True:
        if not sp.isprime(start):
            start+=1
        else:
            return start

def write_pairs_to_file(pairs: np.ndarray) -> None:
    """
    Writes the pairs to a text file.
    Args:
        pairs (np.ndarray): Array of user pairs with their Jaccard similarities
        thresh (float): Similarity threshold for writing pairs
    Returns:
        None
    """
    # Filter pairs exceeding the threshold
    if pairs.size == 0:
        print("No pairs exceed the threshold.")
        return
    # Write only the user numbers to the output file 'results.txt'
    np.savetxt("results.txt", pairs, fmt="%d,%d", delimiter=",")
    print(f"Wrote {pairs.shape[0]} pairs to results.txt.")
