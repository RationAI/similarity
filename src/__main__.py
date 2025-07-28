import ray
import torch
import numpy as np
from src.similarity import compute_similarity, get_top_k_similar


from ray.data.datasource.datasource import Datasource
from src.model import Model


def main(datasource: Datasource, model: torch.nn.Module, k: int = 1):
    df = (
        ray.data.read_datasource(datasource)
        .map_batches(
            Model(model),
            num_gpus=1,
            num_cpus=0,
            batch_size=16,
            memory=3 * 1024 * 1024 * 1024,
            concurrency=1,
        )
        .drop_columns("img")
        .to_pandas()
    )

    # Similarity matrix
    sim_m = compute_similarity(torch.from_numpy(np.array(list(df.features))))

    # Top K Retrieved
    top_k = get_top_k_similar(sim_m, k)

    return df, top_k, sim_m[top_k]
