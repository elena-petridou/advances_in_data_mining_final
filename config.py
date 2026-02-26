"""
Configuration variables for LSH algorithm
"""
import os

if os.path.isfile("./user_movie_rating.npy"):
    PATH = "./user_movie_rating.npy"
else:
    raise FileNotFoundError("""Data file 'user_movie_rating.npy' not found in expected location. Expected in root folder.""")

SEED = 328429
NUM_HASH_FUNCTIONS = 150
NUM_BANDS = 28
THRESHOLD = 0.5
THRESH_BUCKETS = 50
