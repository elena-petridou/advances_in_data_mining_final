"""
Main script for execution of the LSH algorithm
"""

import argparse
import time
import numpy as np
from module import LSH, write_pairs_to_file
from config import PATH, NUM_HASH_FUNCTIONS, NUM_BANDS, THRESH_BUCKETS, SEED
from scipy.sparse import csr_matrix

def load_data(path: str) -> csr_matrix:
    """
    Loads the data from a .npy file into csr_matrix
    Args:
        path (str): Path to the .npy file
    Returns:
        csr_matrix: Sparse matrix representation of the data
    """
    data = np.load(path)
    # Make csr_matrix from numpy array
    rows = data[:, 0].astype(int)
    cols = data[:, 1].astype(int)
    values = data[:, 2]
    csr_data = csr_matrix((values, (rows, cols)))
    return csr_data

def main():
    """
    Main function to run the LSH algorithm with specified parameters. 
    Saves the results to 'results.txt' in the root folder.
    Returns:
        None
    """
    # Create argument parser for seed
    parser = argparse.ArgumentParser("LSH Algorithm", usage='%(prog)s [-h] [-s] [-b] [-f] [-t]')
    parser.add_argument("-s", "--seed",
                    help="Seed for the random number generator used in hash function generation.",
                    metavar='',
                    type=int,
                    default=SEED)
    parser.add_argument("-b", "--bands",
                    help="Number of bands to use in the LSH algorithm.",
                    metavar='',
                    type=int,
                    default=NUM_BANDS)
    parser.add_argument("-f", "--functions",
                    help="Number of hash functions to use in the LSH algorithm.",
                    metavar='',
                    type=int,
                    default=NUM_HASH_FUNCTIONS)
    parser.add_argument("-u", "--users",
                    help="Size limit for users per bucket.",
                    metavar='',
                    type=float,
                    default=THRESH_BUCKETS)
    parser.add_argument("-j", "--jaccard",
                    help="Computes the true Jaccard similarity for the found pairs and returns the true positive rate.",
                    metavar='',
                    type=float,
                    default=True)
    args = parser.parse_args()
    # Load the csr matrix data
    start = time.time()
    df = load_data(PATH)
    # Create instance of LSH class and run the algorithm
    lsh = LSH(data=df)
    lsh.generate_hash_functions(seed=args.seed, num_functions=args.functions)
    lsh.minhashing()
    lsh.band_signatures(num_bands=args.bands)
    lsh.find_candidate_pairs(thresh=args.users)
    lsh.compute_pseudo_jaccard()
    if args.jaccard:
        lsh.compute_real_jaccard()
        print(f"Found {len(lsh.pairs_)} with a pseudo-Jaccard score > 0.5")
        print(f"True positive rate from the pairs with a pseudo-Jaccard similarity > 0.5: {lsh.true_positive_rate_:.2f}")
        write_pairs_to_file(lsh.true_pairs_)
    else:
        write_pairs_to_file(lsh.pairs_)
    total_time = time.time()
    print(f"Total time: {total_time - start:.2f} seconds.")

if __name__ == "__main__":
    main()
