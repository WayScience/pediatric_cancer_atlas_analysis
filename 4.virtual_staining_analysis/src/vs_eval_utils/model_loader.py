"""
model_loader.py

Simple utility function for loading model weights from training runs.
Coupled to the https://github.com/WayScience/virtual_stain_flow 
    virtual_stain_flow package implementation. 
"""

import json
import pathlib

import torch


def load_model_weights(
    run_path: pathlib.Path,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    model_handle: callable = None,
    model_config: dict | None = None,
    compile_model: bool = False,
) -> torch.nn.Module | None:
    """
    Load model weights from a training run directory.

    :param run_path: Path to the training run directory containing logged artifacts.
    :param device: Device to load the model onto (default: CUDA if available, else CPU).
    :param model_handle: Optional callable that returns an uninitialized model instance.
    :param model_config: Optional dictionary of model initialization parameters.
        If not provided, the function will attempt to load a model config from 
        the run artifacts, which may not be available for older logging versions.
    : param compile_model: Whether to compile the model after loading weights (default: False).
    :return: Loaded model with weights, or None if loading failed.
    """

    possible_weight_file_patterns = [
        "generator",
        "model",
        "best",
    ]

    candidate_weight_files = list(
        (run_path / "artifacts").rglob("*.pth")
    )
    matched_files = []
    for pattern in possible_weight_file_patterns:
        matched_files = [file for file in candidate_weight_files if pattern in file.name]
        if matched_files:
            break

    if not matched_files:        
        raise RuntimeError(
            f"No model weight files found in expected locations for run {run_path}."
        )

    # select weights to use
    chosen_weights = matched_files
    # usually best model weights are updated over training and always saved as a
    # specific "best" file
    best_weights = [p for p in chosen_weights if "best" in p.name]
    if best_weights:
        chosen_weight = best_weights[0]
    else:
        # Find the latest weights
        # Assume filenames end with an epoch/step integer
        # Assume model doesn't overtrain with the configured max epochs
        try:
            chosen_weight = sorted(
                chosen_weights,
                key=lambda p: int(p.stem.split("_")[-1]),
            )[-1]
        except ValueError: # allow for non-integer suffixes, fallback to last
            chosen_weight = chosen_weights[-1]
            # all other errors let it fails

    if not chosen_weight.exists():
        raise RuntimeError(
            f"Selected model weight file {chosen_weight} does not exist."
        )    

    if not model_config:
        model_config_path = run_path / "artifacts" / "configs" / "model_config.json"
        if not model_config_path.exists():
            raise RuntimeError(
                "No model config found at expected location "
                f"{model_config_path} for run {run_path}."
                "Please manually provide model_config or ensure logging version saves it."
            )
        try:
            model_config = json.loads(model_config_path.read_text())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model config from {model_config_path}: {e}"
            )
    try:
        model = model_handle(
            **model_config['init']
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize model with config {model_config}: {e}"
        )

    if not model or not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"Model handle did not return a valid torch.nn.Module instance: {model}"
        )
    
    try:
        model.to(device)
        model.eval()
        if compile_model:
            model = torch.compile(model)
        state_dict = torch.load(chosen_weight, map_location=device)
        # print weight loading to see if <all keys matched successfully> message is returned
        print(model.load_state_dict(state_dict))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model weights from {chosen_weight}: {e}"
        )

    return model
