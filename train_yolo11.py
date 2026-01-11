import argparse
import torch
import ultralytics
from ultralytics import YOLO

"""
A sample cli to train yolo11 segmentation:

e.g. run: 
python train_yolo11.py --data Data/Klongchen_YoloDataset/Klongchen_dataset.yaml --model yolo11m-seg.pt --img-size 512 --batch 8 --epochs 100 --cache --amp --workers 12 --name klongchen_v1

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv11 segmentation model"
    )

    parser.add_argument(
        "-d", "--data",
        required=True,
        help="Dataset YAML file (e.g. Klongchen_dataset.yaml)"
    )

    parser.add_argument(
        "-m", "--model",
        default="yolo11n-seg.pt",
        help="Pretrained model (default: yolo11n-seg.pt)"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use: cuda, cpu, or cuda:0 (default: cuda)"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Training image size (default: 512)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader workers (default: 8)"
    )

    parser.add_argument(
        "--project",
        default="runs/segment",
        help="Ultralytics project directory"
    )

    parser.add_argument(
        "--name",
        default="train",
        help="Run name"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume last training run"
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ultralytics.checks()

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {args.device}")

    model = YOLO(args.model, task="segment")

    model.train(
        data=args.data,
        imgsz=args.img_size,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
