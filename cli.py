"""Unified Command-Line Interface for the Road Traffic Sign System.

Usage Examples:
  # Test single image
  python cli.py --source image --input data/samples/class_14_sample_1.png --output out.jpg

  # Test folder of images
  python cli.py --source images --input data/samples/ --output results/output/

  # Test video file
  python cli.py --source video --input data/samples/simulated_driving.mp4 --output results/output/detected.mp4

  # Real-time webcam
  python cli.py --source webcam --camera 0

  # GTSRB Benchmark Evaluation
  python cli.py --source benchmark --split test
"""

import sys
import os
import argparse
from pathlib import Path
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.integration.pipeline import TrafficSignPipeline
from src.config import SAMPLES_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Road Traffic Sign Detection & Recognition System")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["image", "images", "video", "webcam", "benchmark"],
        help="Input modality to process",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to input image, folder, or video")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--cls-conf", type=float, default=0.40, help="Classification confidence threshold")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE environmental enhancement")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index for webcam")
    parser.add_argument("--show", action="store_true", help="Display window during processing")
    parser.add_argument("--backend", type=str, default="keras", choices=["keras", "pytorch"])
    parser.add_argument("--split", type=str, default="test", help="Dataset split for benchmark")

    args = parser.parse_args()

    # Initialize Pipeline
    pipeline = TrafficSignPipeline(classifier_backend=args.backend)

    # 1. Single Image
    if args.source == "image":
        input_path = args.input or str(SAMPLES_DIR / "class_14_sample_1.png")
        out_path = args.output or str(RESULTS_DIR / "output" / f"detected_{Path(input_path).name}")

        print(f"\nProcessing image: {input_path}")
        res = pipeline.process_image(input_path, output_path=out_path, conf_threshold=args.conf)

        print(f"\nResults in {Path(input_path).name} ({res.inference_time_ms} ms):")
        if not res.detections:
            print("  No traffic signs detected.")
        else:
            for i, det in enumerate(res.detections, 1):
                print(
                    f"  [{i}] {det.final_class_name} (Class {det.final_class_id}) | "
                    f"Confidence: {det.final_confidence*100:.2f}% | "
                    f"Box: {det.detection.bbox.to_xyxy()}"
                )
        if res.active_speed_limit is not None:
            print(f"  Active Speed Limit: {res.active_speed_limit} km/h")

        if args.show and res.annotated_frame is not None:
            cv2.imshow("Detection Result", res.annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # 2. Directory of Images
    elif args.source == "images":
        in_dir = Path(args.input or SAMPLES_DIR)
        out_dir = Path(args.output or (RESULTS_DIR / "output"))
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in in_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".ppm"]]
        print(f"\nProcessing {len(images)} images in {in_dir}...")

        total_dets = 0
        for p in images:
            out_file = out_dir / f"det_{p.name}"
            res = pipeline.process_image(str(p), output_path=str(out_file), conf_threshold=args.conf)
            total_dets += len(res.detections)
            print(f"  {p.name}: {len(res.detections)} sign(s) detected ({res.inference_time_ms:.1f} ms)")

        print(f"\nDone. Processed {len(images)} images with {total_dets} total detections.")
        print(f"Saved to: {out_dir}")

    # 3. Video File
    elif args.source == "video":
        input_vid = args.input or str(SAMPLES_DIR / "simulated_driving.mp4")
        out_vid = args.output or str(RESULTS_DIR / "output" / "annotated_video.mp4")

        print(f"\nProcessing video: {input_vid}")
        total = pipeline.process_video(
            input_vid,
            output_path=out_vid,
            show_window=args.show,
            conf_threshold=args.conf,
        )
        print(f"\nVideo processing complete! Total sign detections: {total}")
        print(f"Annotated video saved to: {out_vid}")

    # 4. Live Webcam
    elif args.source == "webcam":
        print(f"\nStarting live webcam stream on index {args.camera} (Press 'q' to exit)...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {args.camera}")
            sys.exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = pipeline.process_frame(frame, conf_threshold=args.conf, render_overlay=True)
            if res.annotated_frame is not None:
                cv2.imshow("Road Traffic Sign Perception — Live Camera", res.annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Camera session closed.")

    # 5. Benchmark Evaluation
    elif args.source == "benchmark":
        from src.classification.evaluate import evaluate_classifier
        evaluate_classifier(split_name=args.split)


if __name__ == "__main__":
    main()
