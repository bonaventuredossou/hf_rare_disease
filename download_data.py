"""Download helpers for the MVA 2026 hackathon dataset on Hugging Face."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

try:
    # Resolves env vars AND the token cached on disk by `hf auth login`.
    from huggingface_hub import get_token as _hub_get_token
except ImportError:  # huggingface_hub < 0.19
    _hub_get_token = None

REPO_ID = "SageBio/mva-hackathon-2026-data"
REPO_TYPE = "dataset"

# Override with RARE_DISEASE_DATA_DIR if you move the data.
DEFAULT_LOCAL_DIR = os.environ.get(
    "RARE_DISEASE_DATA_DIR",
    "/network/scratch/b/bonaventure.dossou/rare_disease_data",
)

_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")

# Organizers state ~85 GB compressed across 11 files, and recommend 100-150 GB
# of headroom for indexing and pipeline intermediates.
COMPRESSED_SIZE_GB = 85.0
RECOMMENDED_FREE_GB = 150.0
IGNORE_PATTERNS = ["README.md", ".gitattributes"]

# Everything below is read from the environment, nothing is hardcoded.
_HF_ENV_VARS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "HF_HUB_OFFLINE",
)


def get_token(required: bool = True):
    """Return the first HF token found in the environment."""
    _, token = token_source()
    if token:
        return token
    if required:
        raise EnvironmentError(
            "No Hugging Face token found. Either run `hf auth login`, "
            "or set one of: " + ", ".join(_TOKEN_ENV_VARS)
        )
    return None


def mask(secret, keep: int = 4) -> str:
    """Render a token safely for logs and notebooks."""
    if not secret:
        return "<unset>"
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}...{secret[-keep:]} (len {len(secret)})"


def token_source():
    """Return (source, token) for the first token found, else (None, None).

    Environment variables win, then the token cached on disk by `hf auth login`
    (under HF_HOME, so it survives across jobs without exporting anything).
    """
    for var in _TOKEN_ENV_VARS:
        token = os.environ.get(var)
        if token:
            return f"${var}", token

    if _hub_get_token is not None:
        token = _hub_get_token()
        if token:
            return f"hf auth login ({hf_home()})", token

    return None, None


def hf_home() -> Path:
    """HF_HOME as set in the environment, or the library default if unset."""
    value = os.environ.get("HF_HOME")
    if value:
        return Path(value).expanduser()
    return Path(os.environ.get("HOME", "~")).expanduser() / ".cache" / "huggingface"


def env_report(local_dir: str = DEFAULT_LOCAL_DIR) -> dict:
    """Print the Hugging Face environment this process actually sees.

    Reads everything through os.environ so nothing is baked into the file.
    """
    var, token = token_source()
    env = {v: os.environ.get(v) for v in _HF_ENV_VARS}

    print("Hugging Face environment")
    print(f"  token          {mask(token)}" + (f"  from {var}" if var else ""))
    for name, value in env.items():
        print(f"  {name:<26} {value if value else '<unset>'}")

    cache = hf_home()
    print(f"  resolved cache             {cache}"
          + ("" if os.environ.get("HF_HOME") else "  (library default)"))
    if cache.exists():
        print(f"  cache free space           {free_space_gb(cache.parent):.1f} GB")

    print(f"  data dir                   {local_dir}")

    if not token:
        print("\n  No token found. Run `hf auth login`, or set one of: "
              + ", ".join(_TOKEN_ENV_VARS))

    home = Path(os.environ.get("HOME", "")).expanduser()
    if home and str(cache).startswith(str(home)):
        print("\n  NOTE: the HF cache sits under $HOME. On a quota'd home directory an "
              "85 GB pull will overrun it. Point HF_HOME at scratch if that applies.")

    return {"token_var": var, **env, "cache": str(cache)}


def ensure_dir(local_dir: str = DEFAULT_LOCAL_DIR, verbose: bool = True) -> Path:
    """Create local_dir if missing and confirm it is a writable directory."""
    path = Path(local_dir).expanduser()

    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"Parent directory does not exist: {parent}. "
            "Check the scratch path before creating anything under it."
        )

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")
        if verbose:
            print(f"Directory exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created directory: {path}")

    if not os.access(path, os.W_OK):
        raise PermissionError(f"Directory is not writable: {path}")

    return path


def free_space_gb(local_dir: str = DEFAULT_LOCAL_DIR) -> float:
    """Free space in GB on the filesystem holding local_dir."""
    path = ensure_dir(local_dir, verbose=False)
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def disk_report(local_dir: str = DEFAULT_LOCAL_DIR) -> float:
    """Print free space on the filesystem holding local_dir. WGS files are large."""
    path = ensure_dir(local_dir, verbose=False)
    stat = os.statvfs(path)
    free = stat.f_bavail * stat.f_frsize / 1e9
    total = stat.f_blocks * stat.f_frsize / 1e9
    print(f"{path}: {free:.1f} GB free of {total:.1f} GB")
    return free


def check_space(
    local_dir: str = DEFAULT_LOCAL_DIR,
    required_gb: float = RECOMMENDED_FREE_GB,
    strict: bool = True,
) -> float:
    """Refuse to start an 85 GB pull that cannot finish.

    Note this reads the filesystem, not your cluster quota. On a quota'd scratch
    the filesystem can look empty while your own allocation is nearly full, so
    check the quota separately.
    """
    free = free_space_gb(local_dir)
    if free < required_gb:
        msg = (
            f"Only {free:.1f} GB free at {local_dir}, "
            f"{required_gb:.0f} GB recommended "
            f"({COMPRESSED_SIZE_GB:.0f} GB compressed plus working room)."
        )
        if strict:
            raise OSError(msg + " Pass strict=False to proceed anyway.")
        print("WARNING: " + msg)
    else:
        print(f"Space OK: {free:.1f} GB free, {required_gb:.0f} GB recommended.")
    return free


def list_files(repo_id: str = REPO_ID, repo_type: str = REPO_TYPE):
    """List everything in the repo so you know what you are pulling."""
    return sorted(list_repo_files(repo_id, repo_type=repo_type, token=get_token()))


def download_file(
    filename: str,
    repo_id: str = REPO_ID,
    repo_type: str = REPO_TYPE,
    local_dir: str = DEFAULT_LOCAL_DIR,
    force: bool = False,
) -> Path:
    """Download one file from the dataset repo into local_dir.

    Skips the network call if the file is already present unless force=True.
    """
    root = ensure_dir(local_dir, verbose=False)
    target = root / filename
    if target.exists() and not force:
        print(f"Already present, skipping: {target}")
        return target

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(root),
            token=get_token(),
            force_download=force,
        )
    )
    print(f"Downloaded {filename} -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def download_snapshot(
    repo_id: str = REPO_ID,
    repo_type: str = REPO_TYPE,
    local_dir: str = DEFAULT_LOCAL_DIR,
    allow_patterns=None,
    ignore_patterns=IGNORE_PATTERNS,
    max_workers: int = 4,
    require_space: bool = True,
    strict_space: bool = True,
) -> Path:
    """Pull the whole challenge dataset (~85 GB, 11 files).

    Resumable: rerun after an interruption and completed files are skipped.
    """
    root = ensure_dir(local_dir)
    if require_space:
        check_space(local_dir, strict=strict_space)

    print(f"Starting snapshot of {repo_id} into {root} with {max_workers} workers.")
    path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(root),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            token=get_token(),
        )
    )
    print(f"\nSnapshot complete: {path}")
    summarize_local(path)
    return path


def summarize_local(local_dir: str = DEFAULT_LOCAL_DIR) -> None:
    """List downloaded files with sizes, largest first, and the total."""
    root = Path(local_dir)
    files = [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts]
    if not files:
        print(f"No files under {root}")
        return
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    total = 0.0
    for p in files:
        gb = p.stat().st_size / 1e9
        total += gb
        print(f"  {gb:8.2f} GB  {p.relative_to(root)}")
    print(f"  {'-' * 8}")
    print(f"  {total:8.2f} GB  total across {len(files)} files")


def download_many(filenames, **kwargs) -> dict:
    """Download several files. Returns {filename: path or None on failure}."""
    results = {}
    for name in filenames:
        try:
            results[name] = download_file(name, **kwargs)
        except Exception as exc:
            print(f"FAILED {name}: {type(exc).__name__}: {exc}")
            results[name] = None
    return results


def download_all(
    repo_id: str = REPO_ID,
    repo_type: str = REPO_TYPE,
    local_dir: str = DEFAULT_LOCAL_DIR,
    suffixes=None,
    **kwargs,
) -> dict:
    """Download every file in the repo, optionally filtered by suffix."""
    files = list_files(repo_id=repo_id, repo_type=repo_type)
    if suffixes:
        files = [f for f in files if f.endswith(tuple(suffixes))]
    return download_many(
        files, repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, **kwargs
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Fetch the MVA 2026 rare disease dataset into scratch."
    )
    ap.add_argument("--dir", default=DEFAULT_LOCAL_DIR)
    ap.add_argument("--list", action="store_true", help="List repo files, download nothing.")
    ap.add_argument("--file", action="append", help="Download one file instead. Repeatable.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-space-check", action="store_true")
    args = ap.parse_args()

    env_report(args.dir)
    print()
    ensure_dir(args.dir)
    disk_report(args.dir)

    if args.list:
        files = list_files()
        print(f"\n{len(files)} files in {REPO_ID}:")
        for f in files:
            print("  ", f)
    elif args.file:
        download_many(args.file, local_dir=args.dir)
    else:
        # Default: pull everything straight into --dir.
        download_snapshot(
            local_dir=args.dir,
            max_workers=args.workers,
            strict_space=not args.no_space_check,
        )