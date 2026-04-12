# Code Assist PR Review Styleguide
**Repository:** `similarity` (RationAI)
**Context:** This is a research-focused machine learning repository implementing a similarity search system for digital pathology whole-slide images (WSIs). It uses deep learning models (GigaPath, Virchow2, UNI2-h, etc.) to extract feature embeddings and computes pairwise similarity to find visually similar tissue samples.

## 🎯 Primary Review Focus
- **Ignore formatting and linting:** Do not comment on line length, quote styles, or basic PEP-8 formatting. Focus on functional correctness.
- **Focus on ML logic and correctness:** Look for off-by-one errors in array slicing, incorrect tensor device allocations, tensor shape mismatches, batch size handling, and GPU memory management issues.
- **Ray pipeline correctness:** Verify that `ray.data` pipelines are configured correctly—check `num_cpus`, `num_gpus`, `batch_size`, and `concurrency` settings for logical consistency.
- **Research context over production validation:** This is research code. Do not suggest adding heavy input validation, complex exception handling, or enterprise-grade defensive programming unless the current logic will explicitly crash the pipeline. Prioritize readability.
- **Testing:** Do not block PRs or aggressively request unit tests. Testing infrastructure is minimal; focus on whether the code works correctly.

## 📝 General Comment Style
- Keep comments **short and actionable**.
- Prefer **bullet points** over long paragraphs.
- Point to specific lines or sections when possible.
- Suggest improvements, not rewrite entire snippets.
- Avoid repetition of what the code already clearly states.
- Defer to the repo's existing conventions unless there's a clear bug or inconsistency.

## 🔬 Domain-Specific Guidance (Digital Pathology & ratiopath)
- **Use `ratiopath`:** This project relies on our library `ratiopath`. 
  - If you see custom WSI reading logic, suggest using `ratiopath.ray.read_slides`.
  - If you see custom tiling logic, suggest using `ratiopath.tiling.grid_tiles` and `read_slide_tiles`.
  - Check if Ray-based distributed processing is being used efficiently for large-scale WSI tasks.
- **WSI Handling:** Verify that `openslide` or `pyvips` calls use the correct downsample levels and that tile offsets are calculated correctly. MPP (microns per pixel) settings must be consistent across the pipeline.
- **Model-Specific Output Handling:** Different encoders return different tensor shapes:
  - **GigaPath:** May return unexpected batch sizes—check for proper shape handling.
  - **Virchow2:** Returns class token + patch tokens; verify pooling/aggregation logic.
  - **UNI2-h / Midnight:** Check embedding dimension consistency.
- **Preprocessing Toggles:** When reviewing preprocessing code (`CPUPreprocessActor`, `TileEncoderActor`), ensure stain normalization, CLAHE, and background removal are applied in the correct order and with consistent data types.

## 🏗️ Architecture & Reproducibility 
- **Command-Line Arguments:** Ensure new CLI arguments are documented in `argparse` help text and are consistent with existing naming conventions (e.g., `--slide-path`, `--save-path`, `--encoder`).
- **Repository Structure:**
  - `src/`: Core library code—review more strictly for type hints and API design.
  - `examples/`: Pipeline scripts and job templates—focus on correctness over perfection.
  - `tests/`: Minimal test coverage; do not expect comprehensive test suites.
  - `pretrained/`: Pre-trained model weights; verify checksums if new weights are added.

## 📚 Types & Documentation
- **Type Hinting:** The codebase uses **heavy type hinting** throughout. Gently suggest adding type hints for new functions, especially for complex function signatures involving DataFrames, tensors, or nested structures. Do not nitpick missing `Any` types.
- **Docstrings:** Docstrings are present on some public APIs (e.g., `gigapathTile()`, `compute_similarity()`). If a docstring *is* provided, ensure it follows the **Google Docstring Style**. Missing docstrings are acceptable for internal/private functions.
- **Comments:** The codebase contains Czech-language comments explaining optimization decisions. While English is preferred for new code, do not block PRs for language inconsistencies. Focus on technical accuracy.

## 💻 Libraries & Best Practices
- **PyTorch:** Watch out for:
  - Detached tensors or memory leaks in custom training steps.
  - Incorrect use of `.to(device)` or `.to(dtype)`—ensure conversions happen before GPU transfer where possible.
  - Use of `torch.no_grad()` during inference.
  - Proper handling of empty batches (e.g., when all tiles are filtered as background).
- **Ray Distributed Processing:** Review resource allocation (`num_cpus`, `num_gpus`, `concurrency`) for appropriateness. Ensure `ActorPoolStrategy` size matches available hardware (e.g., H100 GPU count).
- **Data Processing (NumPy, Pandas, OpenSlide):** Suggest vectorized operations over `for` loops where applicable for performance. Ensure WSI coordinate extractions are logical (e.g., matching the correct level/downsample).
- **GPU Memory Management:** Look for patterns like:
  - Explicit `del` statements after large tensor operations.
  - Use of `float16`/`bfloat16` precision for inference.
  - Chunked/batched processing to fit VRAM constraints.
  - `non_blocking=True` for async CPU→GPU transfers.

## 🚫 What NOT to Comment On
- **Czech vs English comments:** Existing code has Czech comments; do not request translation unless it blocks understanding.
- **Print statements vs logging:** Debug print statements are common in example scripts; only flag if they're excessive or sensitive.
- **Hardcoded paths:** Example scripts often contain hardcoded paths for testing; this is acceptable for research code.
- **Line length violations:** Lines exceeding 100 characters are common; ignore unless they severely impact readability.
- **Import ordering:** Imports are not strictly organized per PEP8; defer to existing patterns.
