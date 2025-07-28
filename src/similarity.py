import torch


def get_top_k_similar(similarity_matrix: torch.tensor, k: int) -> torch.Tensor:
    """
    Finds the top K most similar items for each item in the dataset.

    Args:
        similarity_matrix (np.ndarray): The computed (N, N) cosine similarity matrix.
        k (int): The number of top similar items to retrieve for each item.
    """

    sorted_indices = similarity_matrix.argsort(-1, descending=True)

    # Take top K
    # Ignore first column as it is a self reference
    return sorted_indices[:, 1 : k + 1]


def compute_similarity(features: torch.Tensor, k=1) -> torch.Tensor:
    # Normalize
    features = torch.nn.functional.normalize(features.to(torch.float32), p=2, dim=1)

    # Compute the similarity matrix using matrix multiplication
    return torch.matmul(features, features.T)


# def compute_similarity_from_parquet(directory_path: str, k=1):
#     """
#     Loads all Parquet files from a directory, combines them, and computes
#     the pairwise cosine similarity on the 'features' column.

#     Args:
#         directory_path (str): The path to the directory containing Parquet files.
#     """
#     # Validate that the directory exists
#     if not os.path.isdir(directory_path):
#         print(f"Error: Directory not found at '{directory_path}'")
#         sys.exit(1)

#     # 1. Find all Parquet files in the specified directory
#     # The pattern '*.parquet' matches all files ending with .parquet
#     parquet_files = glob.glob(os.path.join(directory_path, "*.parquet"))

#     if not parquet_files:
#         print(f"Error: No Parquet files found in '{directory_path}'")
#         sys.exit(1)

#     print(f"Found {len(parquet_files)} Parquet files. Loading...")
#     print("\n".join(f"- {os.path.basename(f)}" for f in parquet_files))

#     # 2. Load each Parquet file and store it in a list of DataFrames
#     try:
#         list_of_dfs = [pd.read_parquet(f) for f in parquet_files]
#     except Exception as e:
#         print(
#             "Error reading Parquet files. Make sure you have 'pyarrow' or 'fastparquet' installed."
#         )
#         print("  -> pip install pyarrow")
#         print(f"Original error: {e}")
#         sys.exit(1)

#     # 3. Concatenate all DataFrames into a single one
#     combined_df = pd.concat(list_of_dfs, ignore_index=True)

#     # 4. Check for the 'features' column
#     if "features" not in combined_df.columns:
#         print("Error: The combined DataFrame does not contain a 'features' column.")
#         sys.exit(1)

#     # 5. Extract the features and prepare them for cosine similarity calculation
#     # The cosine_similarity function from scikit-learn can handle a list of lists/arrays.
#     # We convert it to a list for robust processing.
#     print(np.array(combined_df.features), flush=True)

#     features = torch.from_numpy(np.array(combined_df.features)).to(torch.float32)

#     features = torch.nn.functional.normalize(features, p=2, dim=1)
#     # Compute the similarity matrix using matrix multiplication

#     # 6. Compute the pairwise cosine similarity
#     # This returns a NumPy array where the element at (i, j) is the similarity
#     # between the i-th and j-th feature vector.
#     similarity_matrix = torch.matmul(features, features.T)

#     get_top_k_similar(similarity_matrix, combined_df, 3)
