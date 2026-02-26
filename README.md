# LSH Algorithm

## Introduction

This .zip file contains the Python scripts to find users with a > 0.5 Jaccard similarity. To do so, it employs a Locality Sensitive Hashing (LSH) algorithm.

## Content

### `main.py`

Contains the main script to run the algorithm

### `module.py`

Contains the `LSH` class and helper functions

#### `LSH` class

Methods:

* `generate_hash_functions()`: Generates N hash functions to compute MinHash signatures
* `minhashing()`: Computes the MinHash signatures matrix
* `band_signatures()`: Computes bands for the signatures and saves the hash values of each band for later comparison
* `find_candidate_pairs()`: Creates buckets using the banded data and finds users with identical values in a band
* `compute_pseudo_jaccard()`: Computes the pseudo-Jaccard similarity scores using the MinHash signatures for the candidate pairs
* `compute_real_jaccard()`: Computes the true Jaccard similarity score for the pairs that exceed 0.5 in pseudo-Jaccard. Returns the true positive rate

Returns:

* `data_`: The CSR matrix containing the ratings data
* `users_`: The number of users
* `movies_`: The number of movies
* `thresh_`: The threshold for writing pairs to the results textfile
* `hash_functions_`: The generated hash functions
* `num_functions_`: The number of hash functions generated
* `signatures_`: The MinHash signatures
* `num_bands_`: The number of bands for the banding
* `banded_data_`: The banded MinHash signatures
* `candidate_pairs_`: The candidate pairs found after banding. Candidate pairs have at least one band value that is the same
* `similarities_`: The pseudo-Jaccard similarities that exceed 0.5
* `pairs_`: The pairs whose pseudo-Jaccard similarity exceeds 0.5
* `true_similarities_` (optional): The true Jaccard similarities for the pairs with pseudo-Jaccard above 0.5
* `true_pairs_` (optional): The pairs with a true Jaccard similarities above 0.5
* `true_positive_rate_` (optional): The true positive rate of having both pseudo- and true Jaccard above 0.5

#### Helper functions

* `find_next_prime(int)`: Takes a starting number and finds the next prime number
* `write_pairs_to_file(np.ndarray)`: Takes a 2-dimensional array of user pairs and writes them to a textfile in format $u1,u2$

### `config.py`

Sets the default values for LSH parameters and checks for exitence of path to data file.

## Usage

Run the `main.py` script from the terminal. Default parameters are pre-set; however, it is possible to set custom parameters from the command line. Options are below.

### Parameters

| Argument | Description | Default value |
|:---|:---|:---|
| `-s`, `--seed` | Sets the random seed for the generation of hash functions. | 32 |
| `-b`, `--bands`| Sets the  number of bands for the banding of the data. The number of rows per band is inferred. | 24 |
| `-f`, `--functions` | Sets the number of hash function to generate and use during MinHashing | 150 |
| `-u`, `--users` | Sets the limit to the amount of users per bucket. | 150 |
| `-j`, `-jaccard` | Decides whether to compute true Jaccard similarity and return true positive rate of Jaccard similarities > 0.5. If `True`, it writes the pairs with both true Jaccard and pseudo-Jaccard similarity above 0.5 | 1 (`True`) |
