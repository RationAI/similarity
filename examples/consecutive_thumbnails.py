from src.__main__ import main as compute_similarity
from src.feature_extractors import resnet18

from src.datasources import ThumbnailDatasource


def main() -> None:
    paths = [
        "/mnt/data/scans/AI scans/Prostata/Prospective test cases",
    ]

    df, top_k, top_k_sim = compute_similarity(
        ThumbnailDatasource(paths, image_extent=224), resnet18(), k=3
    )

    ref = df["name"].str.extract(r"TP-(.*)\-", expand=False).to_numpy()
    hits = (ref[top_k] == ref.reshape(-1, 1)).any(axis=1)
    recall = hits.mean()

    print(f"Recall: {recall}")


if __name__ == "__main__":
    main()
