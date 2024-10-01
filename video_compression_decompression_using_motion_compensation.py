"""
This script implements motion compensation for video encoding and decoding using macroblocks. 
It reads an input video, processes each frame to detect motion between macroblocks, 
and generates an output video that includes error frames based on the predicted motion. 
Additionally, the script calculates and logs entropy metrics for both original and processed frames, 
providing insights into the compression efficiency.
"""

import argparse
import logging
import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from utils import (
    save_as_gif,
    validate_file_path,
    divide_to_macroblocks,
    create_upper_levels,
    move_to_low_levels,
    motion_exist,
    sad,
    rebuild_image,
    entropy_score,
)


def compress_frames(video_path, output_path):
    """
    Compress frames from the input video using motion compensation.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output error frames video and files.
    """

    validate_file_path(video_path, "Video")

    # Import the original video
    vid = cv2.VideoCapture(video_path)

    # Get video properties: frame count, width, height, and frames per second (fps)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    size = (frameWidth, frameHeight)

    # Create the output video file for error frames
    out = cv2.VideoWriter(
        os.path.join(output_path, "error_frames.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
        False,
    )

    error_frames = []  # List to hold error frames (difference frames)
    original_frames = []  # List to hold original frames
    codec = []  # List to hold predicted and movement blocks

    logging.info("Processing video to extract error frames...")

    ret, prev_frame = vid.read()
    if not ret:
        logging.error("Failed to read the first frame of the video.")
        return

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    error_frames.append(prev_frame)

    with tqdm(total=frameCount, desc="Processing frames", unit="frame") as pbar:
        while vid.isOpened():
            ret, curr_frame = vid.read()

            # Exit the loop if no more frames
            if not ret:
                break

            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            original_frames.append(np.array(curr_frame, dtype="int8"))

            hierimg1, hierimg2 = create_upper_levels(prev_frame, curr_frame)

            # Split images into 4x4 macroblocks at the lowest level (level 3)
            macroblocked_image1 = divide_to_macroblocks(4, hierimg1[2])
            macroblocked_image2 = divide_to_macroblocks(4, hierimg2[2])

            movement_blocks = []

            # Detect movement between macroblocks
            for i in range(len(macroblocked_image1)):
                if motion_exist(macroblocked_image1[i], macroblocked_image2[i]):
                    movement_blocks.append(i)

            # Move back to the lower levels and get the corresponding blocks
            macroblocked_image1, macroblocked_image2, movement_blocks = (
                move_to_low_levels(movement_blocks, hierimg1, hierimg2)
            )

            predicted = []
            for i in range(len(movement_blocks)):
                predicted.append(
                    sad(macroblocked_image1, macroblocked_image2, movement_blocks[i])
                )
                macroblocked_image1[movement_blocks[i]] = macroblocked_image1[
                    predicted[i]
                ]

            codec.append([predicted, movement_blocks])

            # Rebuild the error frame by subtracting predicted blocks from actual blocks
            width = int(frameWidth / 16)
            height = int(frameHeight / 16)
            error_frame = rebuild_image(
                height,
                width,
                np.uint8(np.subtract(macroblocked_image2, macroblocked_image1)),
            )
            out.write(error_frame)
            error_frames.append(error_frame.astype("int8"))

            # Update progress
            pbar.update(1)
            prev_frame = curr_frame

    logging.info("Finished processing video. Saving results...")

    # Calculate entropy for error frames and original frames
    error_entropy = entropy_score(error_frames)
    original_entropy = entropy_score(original_frames)

    # Calculate percentage change in entropy
    entropy_change_percentage = (
        (original_entropy - error_entropy) / original_entropy
    ) * 100

    # Log entropy values and percentage change
    logging.info(f"Error frames entropy: {error_entropy}")
    logging.info(f"Original video entropy: {original_entropy}")
    logging.info(f"Percentage change in entropy: {entropy_change_percentage:.2f}%")

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    # Save results to binary files
    error_frames = np.array(error_frames, dtype="int8")
    specs = np.array([frameCount, frameHeight, frameWidth, fps], dtype="int64")

    error_frames_bin = os.path.join(output_path, "encoded_error_frames.bin")
    error_frames.tofile(error_frames_bin)

    video_specs_bin = os.path.join(output_path, "video_specs.bin")
    specs.tofile(video_specs_bin)

    mov_vectors = os.path.join(output_path, "mov_vectors.bin")

    with open(mov_vectors, "wb") as vectors:
        pickle.dump(codec, vectors)

    # Save error frames as GIF
    gif_path = os.path.join(output_path, "error_frames.gif")
    save_as_gif(error_frames, gif_path)

    logging.info(f"Error frames and specs saved to {output_path}")

    return error_frames_bin, video_specs_bin, mov_vectors


def decompress_frames(encoded_path, specs_path, vectors_path, output_path):
    """
    Decode the frames from the error frame sequence using motion compensation.

    Args:
        encoded_path (str): Path to the encoded frames binary file.
        specs_path (str): Path to the video specs binary file.
        vectors_path (str): Path to the motion vectors binary file.
        output_path (str): Path to save the output decoded video.
    """

    validate_file_path(encoded_path, "Encoded frames")
    validate_file_path(specs_path, "Specs")
    validate_file_path(vectors_path, "Motion vectors")

    # Unpack binary file with the error frames sequence
    logging.info("Loading encoded frames...")
    frames = np.fromfile(encoded_path, dtype="int8")

    # Unpack binary file with the specifications
    logging.info("Loading video specs...")
    specs = np.fromfile(specs_path, dtype="int64")

    # Reshape the frames array to (number of frames, video height, video width)
    frames = np.reshape(frames, (specs[0], specs[1], specs[2]))

    # Load the movement blocks and predicted locations for reconstructing the predicted frames
    logging.info("Loading motion vectors...")
    with open(vectors_path, "rb") as vectors_file:
        codec = pickle.load(vectors_file)

    # Create the output file for decoded video
    out = cv2.VideoWriter(
        os.path.join(output_path, "decoded_video.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        specs[3],
        (specs[2], specs[1]),
        False,
    )

    decoded_frames = []  # List to hold the decoded frames
    logging.info("Decoding video...")

    # Decode frames using motion compensation
    for i, frame in enumerate(frames):
        frame = frame.astype("uint8")

        if i == 0:
            out.write(frame)
            curr_frame = frame
        else:
            macro = divide_to_macroblocks(16, curr_frame)
            macro2 = divide_to_macroblocks(16, curr_frame)

            # Apply motion vectors to reconstruct the image
            for j in range(len(codec[i - 1][1])):
                macro2[codec[i - 1][1][j]] = macro[codec[i - 1][0][j]]

            # Rebuild the image by adding the macroblocks and error frame
            reconstructed_frame = rebuild_image(
                45, 80, np.uint8(np.add(divide_to_macroblocks(16, frame), macro2))
            )

            # Write the decoded frame to the output video
            out.write(reconstructed_frame)
            decoded_frames.append(reconstructed_frame.astype("int8"))

            curr_frame = reconstructed_frame

    logging.info("Decoding complete. Saving results...")

    out.release()
    cv2.destroyAllWindows()

    # Calculate entropy for decoded frames and the original error frames
    decoded_entropy = entropy_score(decoded_frames)
    original_entropy = entropy_score(frames)

    # Calculate percentage change in entropy
    entropy_change_percentage = (
        (decoded_entropy - original_entropy) / original_entropy
    ) * 100

    # Log entropy values and percentage change
    logging.info(f"Decoded video entropy: {decoded_entropy}")
    logging.info(f"Original error frames entropy: {original_entropy}")
    logging.info(f"Percentage change in entropy: {entropy_change_percentage:.2f}%")

    # Save decoded frames as GIF
    gif_path = os.path.join(output_path, "decoded_frames.gif")
    save_as_gif(decoded_frames, gif_path)


def parse_arguments():
    """
    Parses command-line arguments for input/output paths.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Motion compensation video encoding and decoding."
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
        error_frames_bin, specs_bin, mov_vectors = compress_frames(
            args.video_path, args.output_path
        )
        decompress_frames(error_frames_bin, specs_bin, mov_vectors, args.output_path)
        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
