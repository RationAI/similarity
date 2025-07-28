from typing import Any

import numpy as np
import torch


class Model:
    device = "cuda"

    def __init__(self, feature_extractor: torch.nn.Module) -> None:
        self.model = feature_extractor.to(self.device)
        self.model = self.model.eval()

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs = torch.from_numpy(np.array(list(batch["img"]))).to(device=self.device)
        outputs = self.model(inputs).flatten(1)

        batch["features"] = outputs.cpu().numpy()
        return batch
