import argparse
from pathlib import Path

DATASET_ROOT = Path("data/tiblined_yolo")
DEFAULT_MODELS = {"detect": "yolo26n.pt", "semantic": "yolo26n-sem.pt"}


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 on TiBLINED")
    parser.add_argument("--task", choices=DEFAULT_MODELS, default="detect")
    parser.add_argument("--data", type=Path, help="Override the dataset YAML path")
    parser.add_argument(
        "--model", help="Override the pretrained checkpoint or model YAML"
    )
    parser.add_argument(
        "--device", help="CUDA device index, cpu, or mps (auto by default)"
    )
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/tiblined"))
    parser.add_argument("--name", help="Run name (defaults to the task name)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--eval-test", action="store_true", help="Evaluate best.pt on the test split"
    )
    args = parser.parse_args()

    data = (args.data or DATASET_ROOT / f"{args.task}.yaml").resolve()
    if not data.is_file():
        parser.error(f"Dataset YAML not found: {data}. Run prepare_tiblined.py first.")

    from ultralytics import YOLO

    model_path = args.model or DEFAULT_MODELS[args.task]
    model = YOLO(model_path)
    if model.task != args.task:
        parser.error(f"{model_path} is a {model.task!r} model. Expected {args.task!r}.")

    model.train(
        data=str(data),
        imgsz=args.img_size,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.project.resolve()),
        name=args.name or args.task,
        amp=not args.no_amp,
    )

    if args.eval_test:
        best = YOLO(model.trainer.best)
        metrics = best.val(
            data=str(data),
            split="test",
            imgsz=args.img_size,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(model.trainer.save_dir),
            name="test",
        )
        print(metrics.results_dict)


if __name__ == "__main__":
    main()
