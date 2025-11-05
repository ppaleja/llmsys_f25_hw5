"""
Simple Modal wrapper for run_data_parallel.py
Uses a single container with multiple GPUs for distributed training.
"""

import modal

app = modal.App("distributed-training")

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
    .add_local_python_source("project")
    .add_local_python_source("data_parallel")
)

# Shared volume for outputs
volume = modal.Volume.from_name("training-workdir", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G:2",  # Single container with 2 GPUs
    volumes={"/workdir": volume},
    timeout=3600 * 4,
    cpu=16.0,
    memory=65536,
)
def run_training(
    world_size: int = 2,
    dataset: str = "bbaaaa/iwslt14-de-en-preprocess",
    model_max_length: int = 128,
    n_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    pytest: bool = False,
):
    """Run distributed training in a single container with multiple GPUs

    The `pytest` flag will be passed into the training module (by setting
    `project.run_data_parallel.PYTEST`) so the training run can switch into
    pytest mode (short-circuiting long-running evaluation steps).
    """
    import torch.multiprocessing as mp
    import torch.distributed as dist
    import torch

    # Set spawn method before doing anything with CUDA
    mp.set_start_method("spawn", force=True)

    # Import the training module so we can toggle the PYTEST flag before spawning
    import project.run_data_parallel as rdp

    # Ensure the training module sees the pytest flag when run in spawned processes
    import os

    # Set an environment variable so spawned child processes (which inherit the
    # parent environment) will also see the pytest setting. This is necessary
    # when using the 'spawn' start method so the child process modules can read
    # the PYTEST flag during import.
    os.environ["PYTEST"] = "True" if pytest else "False"

    # Set the gradient save directory to the Modal volume root so outputs persist directly
    os.environ["GRADIENT_SAVE_DIR"] = "/workdir"

    rdp.PYTEST = pytest

    # Grab the run_dp function after configuring module-level PYTEST
    run_dp = rdp.run_dp

    processes = []
    backend = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO

    # Spawn processes just like your original code
    for rank in range(world_size):
        p = mp.Process(
            target=run_dp,
            args=(
                rank,
                world_size,
                backend,
                dataset,
                model_max_length,
                n_epochs,
                batch_size,
                learning_rate,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    volume.commit()

    return {"status": "complete", "world_size": world_size, "pytest": pytest}


@app.local_entrypoint()
def main(
    world_size: int = 2,
    dataset: str = "bbaaaa/iwslt14-de-en-preprocess",
    n_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    pytest: bool = False,
):
    """
    Run with: modal run modal_run.py --world-size 2 --n-epochs 5 --pytest True
    """
    print(
        f"Starting distributed training with {world_size} GPUs in single container..."
    )

    result = run_training.remote(
        world_size=world_size,
        dataset=dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pytest=pytest,
    )

    print(f"Training complete: {result}")
