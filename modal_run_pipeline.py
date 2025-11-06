"""
Modal wrapper for project/run_pipeline.py

This mirrors the structure of `modal_run.py` (which wraps `run_data_parallel.py`)
but targets `project/run_pipeline.py`. It:

- Builds a Modal image with the same pinned dependencies used elsewhere in the repo.
- Adds local python sources for `project` and `pipeline` so imports inside
  `project/run_pipeline.py` succeed.
- Exposes a Modal function `run_pipeline` which sets environment variables
  (including `PYTEST` and `GRADIENT_SAVE_DIR`) so spawned/child code can read them.
- Calls `project.run_pipeline.run_pp(...)` with the provided arguments.
- Writes a `TODO.md` into the mounted `/workdir` volume with a short sequential
  log of what this wrapper does (so the user's request to create a TODO is
  recorded in the persistent volume).
"""

import os

import modal

app = modal.App("pipeline-training")

# Define the image with dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",  # Fix numpy version compatibility
        "torch==2.2.0",
        "datasets>=2.4.0,<4.0.0",
        "transformers==4.37.2",
        "sacrebleu==2.4.0",
        "tokenizers",
        "tqdm",
    )
    # Make sure the project modules are available inside the Modal image
    .add_local_python_source("project")
    .add_local_python_source("pipeline")
)

# Shared volume for outputs and artifacts
volume = modal.Volume.from_name("training-workdir", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G:2",  # Single container with up to 2 GPUs for pipeline/model parallelism
    volumes={"/workdir": volume},
    timeout=3600 * 4,
    cpu=16.0,
    memory=65536,
)
def run_pipeline(
    dataset: str = "bbaaaa/iwslt14-de-en-preprocess",
    model_max_length: int = 128,
    n_epochs: int = 2,
    batch_size: int = 32,
    n_chunk: int = 4,
    learning_rate: float = 1e-4,
    model_parallel_mode: str | None = None,
    pytest: bool = False,
):
    """Run the pipeline training script inside Modal.

    Notes:
    - Sets `PYTEST` environment variable and `project.run_pipeline.PYTEST` so
      the pipeline code can switch into short-circuit / pytest mode when needed.
    - Sets `GRADIENT_SAVE_DIR` to `/workdir` so any saved artifacts are placed
      on the mounted volume and persist after the Modal run completes.
    - Writes a `TODO.md` into `/workdir` describing the actions taken.
    """
    # Import locally inside the Modal function to ensure imports happen in the
    # execution environment (image) rather than at script parse time.
    import torch

    # Ensure spawned or imported modules can read the pytest flag
    os.environ["PYTEST"] = "True" if pytest else "False"

    # Ensure outputs are saved to the mounted volume
    os.environ["GRADIENT_SAVE_DIR"] = "/workdir"

    # Import the pipeline runner module and set its module-level PYTEST flag
    import project.run_pipeline as rpp

    # Some modules read PYTEST at import time; set both the env var and the attr
    rpp.PYTEST = pytest

    # Write a short TODO log into the volume root to record the wrapper actions
    todo_lines = [
        "# TODO - modal_run_pipeline wrapper log",
        "",
        "This file was created by `modal_run_pipeline.py` inside the Modal volume",
        "to record the wrapper's setup steps.",
        "",
        "Sequential steps performed:",
        "1. Built a Modal image with pinned dependencies (torch==2.2.0, transformers==4.37.2, ...).",
        "2. Added local python sources: 'project' and 'pipeline'.",
        "3. Mounted a persistent Modal volume at /workdir and set GRADIENT_SAVE_DIR to /workdir.",
        "4. Propagated the `pytest` flag via PYTEST env var and module attribute.",
        "5. Called `project.run_pipeline.run_pp(...)` with provided runtime args.",
        "",
        f"Runtime args passed: dataset={dataset}, model_max_length={model_max_length},",
        f"n_epochs={n_epochs}, batch_size={batch_size}, n_chunk={n_chunk},",
        f"learning_rate={learning_rate}, model_parallel_mode={model_parallel_mode}, pytest={pytest}",
        "",
        "If you want to update this log automatically, modify this wrapper to append",
        "progress information to /workdir/TODO.md at runtime (e.g., each epoch).",
        "",
    ]
    try:
        with open("/workdir/TODO.md", "w") as f:
            f.write("\n".join(todo_lines))
    except Exception:
        # If writing to the volume fails (unexpected), continue; not fatal for the run.
        pass

    # Decide on a device string to pass; run_pipeline.run_pp currently makes its own
    # device choice for MPS vs CPU, but we still pass a value for explicitness.
    device_arg = "cuda" if torch.cuda.is_available() else "cpu"

    # Call the actual pipeline runner
    result = rpp.run_pp(
        dataset_name=dataset,
        model_max_length=model_max_length,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_chunk=n_chunk,
        learning_rate=learning_rate,
        device=device_arg,
        model_parallel_mode=model_parallel_mode,
    )

    # Commit the volume so artifacts are persisted (safe to call even if Modal
    # volume behaviour differs; keeps parity with modal_run.py)
    try:
        volume.commit()
    except Exception:
        # Non-fatal if commit fails in some contexts; still return the run result.
        pass

    return {
        "status": "complete",
        "dataset": dataset,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "model_parallel_mode": model_parallel_mode,
        "pytest": pytest,
    }


@app.local_entrypoint()
def main(
    dataset: str = "bbaaaa/iwslt14-de-en-preprocess",
    model_max_length: int = 128,
    n_epochs: int = 2,
    batch_size: int = 32,
    n_chunk: int = 4,
    learning_rate: float = 1e-4,
    model_parallel_mode: str | None = None,
    pytest: bool = False,
):
    """
    Example invocation:
      modal run modal_run_pipeline.py --dataset bbaaaa/iwslt14-de-en-preprocess --n-epochs 2 --pytest True

    This will submit the `run_pipeline` function to Modal and print the returned
    status dictionary.
    """
    print("Submitting pipeline run to Modal...")

    result = run_pipeline.remote(
        dataset=dataset,
        model_max_length=model_max_length,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_chunk=n_chunk,
        learning_rate=learning_rate,
        model_parallel_mode=model_parallel_mode,
        pytest=pytest,
    )

    print(f"Modal run returned: {result}")
