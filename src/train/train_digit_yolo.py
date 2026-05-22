from ultralytics import YOLO
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="digit_yolo26n_v1")
    parser.add_argument("--predict", type=str, default=None)

    args = parser.parse_args()
    args.data = str(Path(args.data).expanduser().resolve())

    model = YOLO(args.model)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        augment=True,

        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=7.0,
        translate=0.12,
        scale=0.5,
        shear=2.5,
        perspective=0.0005,

        mosaic=1.0,
        close_mosaic=20,
        mixup=0.5,
        copy_paste=0.25,

        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        patience=20,
        cos_lr=True,

        workers=4,
        plots=True,
        save=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weight = save_dir / "weights" / "best.pt"

    print(f"\n학습 완료: {best_weight}")

    model = YOLO(str(best_weight))

    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    if args.predict:
        model.predict(
            source=args.predict,
            imgsz=args.imgsz,
            conf=0.25,
            iou=0.45,
            device=args.device,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(save_dir.parent),
            name=f"{save_dir.name}_predict",
        )


if __name__ == "__main__":
    main()