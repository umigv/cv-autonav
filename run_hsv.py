import cv2
import numpy as np
import time
import os

# Assuming your class code above is in a file named hsv_processor.py
# If it's in the same file, you can remove this import line.
from hsv import hsv 

def benchmark_hsv_masking(video_path, barrel_mode="YOLO", yolo_lanes=False, yolo_barrels=False):
    """
    Benchmarks the performance of the hsv.get_mask() pipeline on a given video.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # 1. Initialize the hsv library instance
    print("Initializing HSV and YOLO models...")
    processor = hsv(video_path=video_path)
    processor.set_YOLO_lanes(yolo_lanes)
    processor.set_YOLO_barrels(yolo_barrels)

    # 2. Open the video capture stream
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Target Video: {video_path}")
    print(f"Total Frames to Process: {total_video_frames} | Native FPS: {fps_source:.2f}")
    print("--------------------------------------------------")
    print("Benchmarking pipeline processing loop...")

    frame_times = []
    processed_count = 0

    # Start overall tracking clock
    total_start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Video ended

        # Start precise frame timer
        start_frame = time.perf_counter()
        
        # --- CORE TARGET FUNCTION EXECUTION ---
        # Calls get_mask, which runs gamma adjustment, color conversions,
        # contour filtering, and conditional YOLO inferences.
        combined_mask, individual_masks = processor.get_mask(frame)
        # --------------------------------------
        
        # Stop precise frame timer
        end_frame = time.perf_counter()
        
        # Record delta time in milliseconds
        elapsed_ms = (end_frame - start_frame) * 1000.0
        frame_times.append(elapsed_ms)
        processed_count += 1

        # Provide a live console ticker so you know it hasn't frozen
        if processed_count % 30 == 0 or processed_count == total_video_frames:
            print(f"Processed Frame {processed_count}/{total_video_frames} "
                  f"(Latest Latency: {elapsed_ms:.2f} ms)", end="\r")

    # Final overall loop time tracking
    total_end_time = time.perf_counter()
    cap.release()
    print("\nProcessing complete! Generating metrics...")
    print("--------------------------------------------------")

    # 3. Compute Benchmark Statistics
    if processed_count == 0:
        print("No frames were processed.")
        return

    total_pipeline_time_sec = total_end_time - total_start_time
    avg_latency_ms = np.mean(frame_times)
    median_latency_ms = np.median(frame_times)
    min_latency_ms = np.min(frame_times)
    max_latency_ms = np.max(frame_times)
    std_dev_ms = np.std(frame_times)
    
    # Calculate effective processing frames-per-second (FPS)
    effective_fps = processed_count / total_pipeline_time_sec

    # 4. Print Benchmark Summary Reports
    print(f"## BENCHMARK RESULTS FOR: {os.path.basename(video_path)} ##")
    print(f"Active Features  | YOLO Lanes: {yolo_lanes} | YOLO Barrels: {yolo_barrels} (Mode: {barrel_mode})")
    print(f"Frames Processed | {processed_count} frames")
    print(f"Total Wall Time  | {total_pipeline_time_sec:.3f} seconds")
    print(f"Effective Speed  | {effective_fps:.2f} FPS")
    print("--- Latency Metrics (get_mask processing time only) ---")
    print(f"Average Latency  | {avg_latency_ms:.2f} ms per frame")
    print(f"Median Latency   | {median_latency_ms:.2f} ms per frame")
    print(f"Standard Dev     | {std_dev_ms:.2f} ms")
    print(f"Min / Max Range  | [{min_latency_ms:.2f} ms - {max_latency_ms:.2f} ms]")
    print("--------------------------------------------------")


if __name__ == "__main__":
    # --- CONFIGURATION PROFILE ---
    # Replace with your actual benchmark test assets
    TEST_VIDEO = "pothole_lane_change.mp4" 
    
    # Configure parameter flags to see exactly how big of a performance impact
    # activating your YOLOv8 background models has on performance.
    RUN_WITH_YOLO_LANES = False
    RUN_WITH_YOLO_BARRELS = False
    BARREL_DETECTION_MODE = "YOLO"  # "YOLO" or custom filter name string
    # -----------------------------

    benchmark_hsv_masking(
        video_path=TEST_VIDEO,
        barrel_mode=BARREL_DETECTION_MODE,
        yolo_lanes=RUN_WITH_YOLO_LANES,
        yolo_barrels=RUN_WITH_YOLO_BARRELS
    )