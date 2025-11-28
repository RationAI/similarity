from kube_jobs import storage, submit_job
import hydra

@hydra.main(version_base="1.2", config_path="./", config_name="config.yaml")
def main(cfg):
    submit_job(
        job_name=cfg.job.job_name,
        username=cfg.job.username,
        cpu=cfg.job.cpu,
        shm=cfg.job.shm,
        memory=cfg.job.memory,
        gpu=cfg.job.gpu,
        public=cfg.job.public,
        script=[
        "export HF_TOKEN=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",
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/tools/similarity.git",
        "cd similarity",
        "git checkout feature/slide-encoder",
        "uv sync",
        "source .venv/bin/activate",
        f"python -m examples.slide_embeddings --slide-path {cfg.slide.slide_path} --save-path {cfg.slide.save_path} --overwrite {cfg.slide.overwrite} --batch-size {cfg.workers.batch_size} --workers {cfg.workers.workers}",
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )

if __name__ == "__main__":
    main()