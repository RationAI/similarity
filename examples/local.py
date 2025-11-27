from examples.slide_embeddings import process_slide
import hydra
import torch
import time
import os

@hydra.main(version_base="1.2", config_path="./", config_name="config.yaml")
def main(cfg):
    slide_path = cfg.slide.slide_path
    save_path = cfg.slide.save_path
    OVERRIDE= cfg.slide.overwrite
    NUM_WORKERS = cfg.workers.workers
    BATCH_SIZE = cfg.workers.batch_size

    MODEL_DTYPE = torch.bfloat16
    DEVICE = torch.device("cuda")
    SLIDE_COUNT = 0



    # disclaimer: based on testing, can be wrong
    MODEL_SIZE_GB = 4.8
    ONE_BATCH_SIZE_GB = 0.0113  # size of 1 tile 256x256
    OVERHEAD = 2 # for pytorch and os stuff

    VRAM_PER_WORKER = ((BATCH_SIZE * ONE_BATCH_SIZE_GB) + MODEL_SIZE_GB)

    TOTAL_VRAM = torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3

    if (NUM_WORKERS == 0):
        NUM_WORKERS = int((TOTAL_VRAM) // VRAM_PER_WORKER)
        if NUM_WORKERS == 0:
            NUM_WORKERS = 1

    print(f"Number of workers: {NUM_WORKERS:.2f}")
    print(f"total vram: {TOTAL_VRAM:.2f}")
    print(f"vram per worker: {VRAM_PER_WORKER:.2f}")

    start_time = time.time()
    if os.path.isdir(slide_path):
        for slide_name in os.listdir(slide_path):
            absolute_path = os.path.join(slide_path, slide_name)
            if os.path.isdir(absolute_path) or absolute_path.endswith(".xml"):
                continue
            SLIDE_COUNT += 1
            process_slide(absolute_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE)
    else:
        process_slide(slide_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE)
        SLIDE_COUNT += 1
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"=============================================")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Slide processed: {SLIDE_COUNT} slides")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"total vram: {TOTAL_VRAM:.2f}")
    print(f"vram per worker: {VRAM_PER_WORKER:.2f}")
    print(f"=============================================")

if __name__ == "__main__":
    main()