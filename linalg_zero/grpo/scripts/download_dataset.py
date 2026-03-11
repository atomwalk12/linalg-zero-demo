from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="atomwalk12/LinalgZero-SFT-110",
    repo_type="model",
    local_dir="./downloaded_best_models",
    allow_patterns=["*"],
)
