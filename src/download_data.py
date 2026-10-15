"""Member A: download_data module entry point."""
from src.data_pipeline.download_data import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download or link GTSRB dataset")
    parser.add_argument("--path", type=str, default=None, help="Path to local extracted GTSRB dataset")
    args = parser.parse_args()
    download_or_link_dataset(args.path)
