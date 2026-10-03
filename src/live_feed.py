"""
Continuous Real-Time Driving Stream Runner (Dashcam / Webcam / Video Stream).
Maintained by Member D (Integration & Pipeline Lead).
Simulates real-time vehicle driver assistance with temporal tracking and speed limit display.
"""

import argparse
import time
import os
import cv2
import numpy as np

from src.pipeline import TrafficSignPipeline


def run_continuous_stream(
    source: str | int = 0,
    mode: str = "heuristic",
    conf_threshold: float = 0.45,
    record_path: str = None,
    show_window: bool = True
):
    print(f"\n========================================================")
    print(f"[ADAS] Continuous Traffic Sign Detection & Tracking")
    print(f"========================================================")
    print(f"Source: {source} | Pipeline Mode: {mode} | Conf: {conf_threshold}")
    print(f"Keys: [Q] Quit | [P] Pause/Resume | [S] Save Snapshot\n")

    # If source is digit string like "0", convert to integer
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        return

    # Initialize Pipeline with Temporal Tracking enabled
    pipeline = TrafficSignPipeline.create(
        mode=mode,
        default_conf_threshold=conf_threshold,
        enable_tracking=True
    )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    writer = None
    if record_path:
        out_dir = os.path.dirname(record_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(record_path, fourcc, input_fps, (width, height))
        print(f"Recording continuous output to: {record_path}")

    frame_count = 0
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[Info] End of video stream reached.")
                    break

                frame_count += 1
                t0 = time.perf_counter()
                result = pipeline.process_frame(frame, conf_threshold=conf_threshold)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                annotated = result.annotated_frame

                if writer:
                    writer.write(annotated)

                # Terminal log update
                active_speed = result.active_speed_limit or "--"
                active_haz = result.active_hazard or "None"
                print(
                    f"\rFrame #{frame_count:04d} | "
                    f"FPS: {result.fps:4.1f} | "
                    f"Latency: {result.total_latency_ms:4.1f}ms | "
                    f"Visible Signs: {result.num_signs_detected:1d} | "
                    f"Active Speed: {active_speed:7s} | "
                    f"Hazard: {active_haz}",
                    end="",
                    flush=True
                )

                if show_window:
                    cv2.imshow("ADAS Traffic Sign Detection (Continuous Video)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[Info] Streaming terminated by user.")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"\n[Info] Stream {'Paused' if paused else 'Resumed'}.")
            elif key == ord("s"):
                snap_name = f"snapshot_frame_{frame_count}.jpg"
                cv2.imwrite(snap_name, annotated)
                print(f"\n[Info] Saved screenshot: {snap_name}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Continuous Traffic Sign Stream Runner")
    parser.add_argument("--source", "-s", default="0", help="Webcam index (0) or path to video file (.mp4)")
    parser.add_argument("--mode", "-m", default="heuristic", choices=["mock", "heuristic", "production"])
    parser.add_argument("--conf", "-c", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--record", "-r", default=None, help="Optional output path to record video (.mp4)")
    parser.add_argument("--no-window", action="store_true", help="Run headless without cv2.imshow")
    args = parser.parse_args()

    run_continuous_stream(
        source=args.source,
        mode=args.mode,
        conf_threshold=args.conf,
        record_path=args.record,
        show_window=not args.no_window
    )


if __name__ == "__main__":
    main()
