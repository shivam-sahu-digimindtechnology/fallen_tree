import cv2
import sys
from config import settings
from detector import FallenTreeDetector
from bounding_box import draw_bounding_boxes, print_detections


def main():
    print("=" * 60)
    print("  Fallen Tree Detection System")
    print("=" * 60)
    print(f"Model        : {settings.MODEL_PATH}")
    print(f"Video source : {settings.VIDEO_SOURCE}")
    print(f"Confidence   : {settings.CONFIDENCE_THRESHOLD}")
    print("-" * 60)

    try:
        detector = FallenTreeDetector()
    except Exception as e:
        print(f"[FATAL] Failed to initialize detector: {e}")
        sys.exit(1)

    source = settings.VIDEO_SOURCE
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[FATAL] Cannot open video source: {source}")
        print("Tip: Set VIDEO_SOURCE = 0 in config.py for webcam")
        sys.exit(1)

    frame_idx = 0
    writer = None
    total_detections = 0

    if settings.SAVE_OUTPUT:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(settings.OUTPUT_PATH, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {settings.OUTPUT_PATH}")

    print("\n[INFO] Starting detection... Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video / stream.")
                break

            frame_idx += 1

            # Detect
            detections = detector.detect(frame)
            total_detections += len(detections)

            if detections and settings.PRINT_DETECTIONS:
                print_detections(detections, frame_idx)

            # Draw boxes + alert banner
            annotated = draw_bounding_boxes(frame, detections)

            # Frame counter
            status = f"Frame: {frame_idx}"
            cv2.putText(
                annotated,
                status,
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            if settings.SHOW_WINDOW:
                cv2.imshow("Fallen Tree Detection", annotated)

            if writer is not None:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Quit by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        print("-" * 60)
        print(f"Processed frames     : {frame_idx}")
        print(f"Total detections     : {total_detections}")
        print("Done.")
        print("=" * 60)


if __name__ == "__main__":
    main()