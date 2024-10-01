"""
This script implements an encoder and decoder for error frames in a video. 
It takes an input video, processes the difference between consecutive frames 
(P-frames) without motion compensation, and calculates the sequence of error images. 
The error images are then saved and encoded in a binary format. The decoder reconstructs 
the video using the encoded error frames, and the script also computes the entropy of the 
original and error frames to highlight the difference in information content.
"""

import os
import cv2
import numpy as np
import argparse
import logging
from utils import validate_file_path, entropy_score, save_as_gif


def extract_error_frames(video_path, output_path):
    """
    Extracts error frames from a video, saves them, and encodes them in binary format.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the processed video and binary files.

    Returns:
        tuple: Paths to the error frames binary and specs binary.
    """
    validate_file_path(video_path, "Video")

    # Capture the video
    vid = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Set output video file
    size = (frame_width, frame_height)
    error_video_out = cv2.VideoWriter(
        os.path.join(output_path, "error_frames.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
        False,
    )

    error_frames = []
    original_frames = []

    logging.info("Processing video to extract error frames...")

    # Read first frame
    ret, prev_frame = vid.read()
    if not ret:
        raise ValueError("Error reading the first frame of the video.")

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    error_frames.append(np.array(prev_frame, dtype="int8"))
    original_frames.append(np.array(prev_frame, dtype="int8"))

    # Loop through frames to calculate error frames
    while vid.isOpened():
        ret, curr_frame = vid.read()
        if not ret:
            break

        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        original_frames.append(np.array(curr_frame, dtype="int8"))

        error_frame = np.subtract(curr_frame, prev_frame)
        error_video_out.write(error_frame)

        error_frames.append(error_frame.astype("int8"))
        prev_frame = curr_frame

    logging.info("Error frames processed. Calculating entropy...")

    original_entropy = entropy_score(original_frames)
    error_entropy = entropy_score(error_frames)
    entropy_change_percentage = (
        ((original_entropy - error_entropy) / original_entropy) * 100
        if original_entropy != 0
        else 0
    )

    # Log entropy values for comparison
    logging.info(f"Entropy of error frames: {error_entropy}")
    logging.info(f"Entropy of original frames: {original_entropy}")
    logging.info(f"Percentage change in entropy: {entropy_change_percentage:.2f}%")

    vid.release()
    error_video_out.release()
    cv2.destroyAllWindows()

    # Save error frames and video specs to binary files
    error_frames_np = np.array(error_frames)
    specs_np = np.array([frame_count, frame_height, frame_width, fps], dtype="int64")

    error_frames_bin = os.path.join(output_path, "encoded_error_frames.bin")
    specs_bin = os.path.join(output_path, "video_specs.bin")
    error_frames_np.tofile(error_frames_bin)
    specs_np.tofile(specs_bin)

    # Save error frames as GIF
    gif_path = os.path.join(output_path, "error_frames.gif")
    save_as_gif(error_frames_np, gif_path)

    logging.info(f"Error frames and specs saved to {output_path}")

    return error_frames_bin, specs_bin


def decode_error_frames(encoded_error_path, specs_path, output_path):
    """
    Decodes error frames from binary files and reconstructs the original video.

    Args:
        encoded_error_path (str): Path to the encoded error frames binary file.
        specs_path (str): Path to the video specs binary file.
        output_path (str): Path to save the reconstructed video.
    """
    logging.info("Decoding error frames...")

    # Load encoded frames and specs
    error_frames = np.fromfile(encoded_error_path, dtype="int8")
    specs = np.fromfile(specs_path, dtype="int64")

    # Reshape error frames into 3D array
    frame_count, frame_height, frame_width, fps = specs
    frames = np.reshape(error_frames, (frame_count, frame_height, frame_width))

    # Create the output video file
    video_out = cv2.VideoWriter(
        os.path.join(output_path, "reconstructed_video.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (frame_width, frame_height),
        False,
    )

    first_frame_flag = True
    reconstructed_frames = []
    for frame in frames:
        frame = frame.astype("uint8")
        if first_frame_flag:
            video_out.write(frame)
            reconstructed_frames.append(frame)
            prev_frame = frame
            first_frame_flag = False
        else:
            image = np.add(prev_frame, frame).astype("uint8")
            video_out.write(image)
            reconstructed_frames.append(image)
            prev_frame = image

    video_out.release()
    cv2.destroyAllWindows()

    # Save reconstructed video as GIF
    gif_path = os.path.join(output_path, "reconstructed_video.gif")
    save_as_gif(reconstructed_frames, gif_path)

    logging.info(f"Reconstructed video saved to {output_path}")


def parse_arguments():
    """
    Parses command-line arguments for input/output paths.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process and decode error frames from a video."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="OriginalVideos/lion.mp4",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory for saving files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level="INFO"
    )

    os.makedirs(args.output_path, exist_ok=True)

    try:
        logging.info("Starting video processing...")
        error_frames_bin, specs_bin = extract_error_frames(
            args.video_path, args.output_path
        )
        decode_error_frames(error_frames_bin, specs_bin, args.output_path)
        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
