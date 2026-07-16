import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BACKEND_DIR / "local_models"

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def model_folder_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def download_model(model_id: str, target_root: Path) -> Path:
    target_dir = target_root / model_folder_name(model_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} to {target_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
    )
    return target_dir


def detect_hidden_size(model_dir: Path) -> int:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "dim", None)
    if hidden_size is None:
        raise RuntimeError(f"Could not detect embedding dimension from {model_dir}")
    return int(hidden_size)


def update_env(path: Path, updates: dict) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen = set()
    next_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            next_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            next_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            next_lines.append(line)

    if next_lines and next_lines[-1].strip():
        next_lines.append("")

    for key, value in updates.items():
        if key not in seen:
            next_lines.append(f"{key}={value}")

    path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hugging Face models for local RAG inference.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--env-file", default=str(BACKEND_DIR / ".env"))
    args = parser.parse_args()

    model_root = Path(args.model_dir).resolve()
    env_file = Path(args.env_file).resolve()

    embed_dir = download_model(args.embed_model, model_root)
    chat_dir = download_model(args.chat_model, model_root)
    embed_dim = detect_hidden_size(embed_dir)

    update_env(
        env_file,
        {
            "INFERENCE_PROVIDER": "local",
            "EMBEDDING_PROVIDER": "local",
            "EMBED_MODEL": args.embed_model,
            "EMBED_MODEL_PATH": str(embed_dir),
            "EMBED_DIM": str(embed_dim),
            "CHAT_MODEL": args.chat_model,
            "CHAT_MODEL_PATH": str(chat_dir),
            "ALLOW_MODEL_DOWNLOADS": "false",
        },
    )

    print("")
    print("Local model setup complete.")
    print(f"Embedding model: {args.embed_model}")
    print(f"Embedding path:  {embed_dir}")
    print(f"Embedding dim:   {embed_dim}")
    print(f"Chat model:      {args.chat_model}")
    print(f"Chat path:       {chat_dir}")
    print(f"Updated env:     {env_file}")
    print("")
    print("Restart the backend after this change.")


if __name__ == "__main__":
    main()
