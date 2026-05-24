import sys

from ml_training.train_deepfake_efficientnet import main


if __name__ == "__main__":
    if "--architecture" not in sys.argv:
        sys.argv[1:1] = ["--architecture", "convnext_tiny"]
    raise SystemExit(main())
