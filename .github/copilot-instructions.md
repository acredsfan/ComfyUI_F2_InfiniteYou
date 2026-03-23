# Copilot Instructions for ComfyUI_F2_InfiniteYou

## Project Overview

**ComfyUI_F2_InfiniteYou** is a ComfyUI custom node pack that provides FLUX-compatible identity-preserved image generation using the [InfiniteYou](https://github.com/bytedance/InfiniteYou) framework by ByteDance. It is intentionally namespaced as `F2_` to coexist alongside the original `ComfyUI_InfiniteYou` pack without collision.

- All ComfyUI node class names are prefixed with `F2_`.
- All nodes are grouped under the `f2_infinite_you` category.
- Model files are read from `ComfyUI/models/infinite_you/`.

## Repository Structure

```
.
├── __init__.py          # Registers nodes with ComfyUI
├── nodes.py             # All ComfyUI node definitions
├── infuse_net.py        # InfuseNet (ControlNet-style) model loading and FLUX patching
├── resampler.py         # Resampler / image projection model architecture
├── utils.py             # Shared utility functions (image ops, face embedding, etc.)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata and Comfy Registry config
└── examples/            # Example workflows and images
```

## Key Nodes

| Node Class | Display Name | Purpose |
|---|---|---|
| `F2_IDEmbeddingModelLoader` | F2 ID Embedding Model Loader | Loads InsightFace detector, ArcFace model, and image projection model |
| `F2_ExtractIDEmbedding` | F2 Extract ID Embedding | Extracts identity embedding from a face image |
| `F2_ExtractFacePoseImage` | F2 Extract Face Pose Image | Extracts face keypoints overlay for pose control |
| `F2_InfuseNetLoader` | F2 Load InfuseNet | Loads the InfuseNet (FLUX ControlNet) checkpoint |
| `F2_InfuseNetApply` | F2 Apply InfuseNet | Applies identity + pose conditioning to a FLUX pipeline |

## Dependencies and Environment

- **Python ≥ 3.10** (follows ComfyUI requirements)
- **PyTorch** with CUDA support (installed by ComfyUI)
- Key packages: `facexlib`, `insightface`, `onnxruntime`, `opencv-python`, `huggingface_hub`
- Install dependencies: `pip install -r requirements.txt`
- This pack runs **inside ComfyUI** as a custom node; `folder_paths` and `comfy` are provided by the ComfyUI runtime.

## Coding Conventions

- **License header**: Every `.py` file must start with the Apache 2.0 license header (see existing files).
- **Typing**: Keep type hints minimal to match the existing style; avoid adding heavy type annotation where not present.
- **Torch dtype**: Model inference uses `torch.bfloat16`; maintain this unless there is a documented reason to change it.
- **Device handling**: Always obtain the compute device via `comfy.model_management.get_torch_device()`.
- **Node structure**: Each node class must define `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, and `CATEGORY` class attributes following the ComfyUI node API.
- **Error messages**: Raise `ValueError` with a clear, user-facing message when inputs are invalid (see `_validate_flux_compatibility` in `nodes.py` for style reference).
- **Model downloads**: Use `huggingface_hub.hf_hub_download` or `snapshot_download` for auto-downloading models at runtime; always check existence before downloading.
- **No global state**: Avoid module-level mutable state; keep model instances scoped to node method calls or class instances.

## Testing

There is no automated test suite. Validate changes by:

1. Installing the pack in a working ComfyUI environment.
2. Loading the example workflow from `examples/infinite_you_workflow.json`.
3. Running the workflow end-to-end to confirm no regressions.

When writing new utility functions in `utils.py`, keep them pure (no side effects) so they can be tested independently with `pytest` if a test suite is added later.

## FLUX Compatibility Notes

- The nodes support the full FLUX family (FLUX.1, FLUX.2). Missing pooled embeddings are handled with zero-tensor fallbacks.
- Dimension mismatches between the image projection model and InfuseNet are detected early in `_validate_flux_compatibility` and surfaced as clear errors.
- `guidance` (distillation) embedding is forwarded if present, otherwise a zero tensor is used.
- Do not hard-code hidden dimensions; derive them from the loaded checkpoint (see `_infer_linear_input_features` in `infuse_net.py`).

## Pull Request Guidelines

- Keep changes focused and minimal.
- Update `README.md` if user-visible behavior changes (new nodes, changed parameters, new model support).
- Do not commit model weight files or large binaries; only code and configuration files belong in the repo.
- Increment `version` in `pyproject.toml` when publishing a release.
