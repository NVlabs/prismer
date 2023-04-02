from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import disable_progress_bars
from pathlib import Path
from rich.progress import Progress
from fire import Fire
from typing import Union, List

_EXPERTS = [
    "10_model.pth",
    "Unified_learned_OCIM_RS200_6x+2x.pth",
    "dpt_hybrid-midas-501f0c75.pt",
    "icdar2015_hourglass88.pth",
    "model_final_e0c58e.pkl",
    "model_final_f07440.pkl",
    "scannet.pt",
]

_MODELS = [
    "vqa_prismer_base",
    "vqa_prismer_large",
    "vqa_prismerz_base",
    "vqa_prismerz_large",
    "caption_prismerz_base",
    "caption_prismerz_large",
    "caption_prismer_base",
    "caption_prismer_large",
    "pretrain_prismer_base",
    "pretrain_prismer_large",
    "pretrain_prismerz_base",
    "pretrain_prismerz_large",
]

_REPO_ID = "shikunl/prismer"


def download_checkpoints(
        download_experts: bool = False,
        download_models: Union[bool, List] = False,
        hide_tqdm: bool = False,
        force_redownload: bool = False,
):
    if hide_tqdm:
        disable_progress_bars()
    # Convert to list and check for invalid names
    download_experts = _EXPERTS if download_experts else []
    if download_models:
        # only download single model
        if isinstance(download_models, str):
            download_models = [download_models]
        # download all models
        if isinstance(download_models, bool):
            download_models = _MODELS

        assert all([m in _MODELS for m in download_models]), f"Invalid model name. Must be one of {_MODELS}"
    else:
        download_models = []

    # Check if files already exist
    if not force_redownload:
        download_experts = [e for e in download_experts if not Path(f"./experts/expert_weights/{e}").exists()]
        download_models = [m for m in download_models if not Path(f"{m}/pytorch_model.bin").exists()]

    assert download_experts or download_models, "Nothing to download."

    with Progress() as progress:
        # Calculate total download size
        progress.print("[blue]Calculating download size...")
        total_size = 0
        for expert in download_experts:
            url = hf_hub_url(
                filename=expert,
                repo_id=_REPO_ID,
                subfolder="expert_weights"
            )
            total_size += get_hf_file_metadata(url).size

        for model in download_models:
            url = hf_hub_url(
                filename=f"pytorch_model.bin",
                repo_id=_REPO_ID,
                subfolder=model
            )
            total_size += get_hf_file_metadata(url).size
        progress.print(f"[blue]Total download size: {total_size / 1e9:.2f} GB")

        # Download files
        total_files = len(download_experts) + len(download_models)
        total_task = progress.add_task(f"[green]Downloading files", total=total_files)
        if download_experts:
            expert_task = progress.add_task(
                f"[green]Downloading experts...", total=len(download_experts)
                )
            out_folder = Path("experts/expert_weights")
            out_folder.mkdir(parents=True, exist_ok=True)
            for expert in download_experts:
                path = Path(hf_hub_download(
                    filename=expert,
                    repo_id=_REPO_ID,
                    subfolder="expert_weights"
                ))
                path.resolve().rename(out_folder/path.name)
                path.unlink()
                progress.advance(expert_task)
                progress.advance(total_task)

        if download_models:
            model_task = progress.add_task(
                f"[green]Downloading models...", total=len(download_models)
                )
            for model in download_models:
                path = Path(hf_hub_download(
                    filename=f"pytorch_model.bin",
                    repo_id=_REPO_ID,
                    subfolder=model
                ))
                out_folder = Path("./logging")/model
                out_folder.mkdir(parents=True, exist_ok=True)
                path.resolve().rename(out_folder/"pytorch_model.bin")
                path.unlink()
                progress.advance(model_task)
                progress.advance(total_task)
        progress.print("[green]Done!")


if __name__ == "__main__":
    Fire(download_checkpoints)
