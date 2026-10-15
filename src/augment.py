"""Member A: augment module entry point."""
from src.data_pipeline.augment import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="Target images per class")
    args = parser.parse_args()
    balance_and_augment(target_per_class=args.target)
