import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download BDRC/TiBLINED dataset"
    )
    parser.add_argument("--output", type=Path, default=Path("data/TiBLINED"))
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or commit")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalTokenNotFoundError, RepositoryNotFoundError

    try:
        path = snapshot_download(
            repo_id="BDRC/TiBLINED",
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output,
            token=True,
        )
    except (LocalTokenNotFoundError, RepositoryNotFoundError):
        parser.exit(
            1,
            "Sign in with an account that can read BDRC/TiBLINED using `hf auth login`, then retry.\n",
        )
    print(path)


if __name__ == "__main__":
    main()
