"""
This script performs motion compensation on videos to remove moving objects 
using macroblock analysis and hierarchical image processing. It saves the output 
video and generates a GIF of the processed frames. The script utilizes OpenCV 
and NumPy for video manipulation and numerical operations.
"""

import numpy as np
import cv2
import os
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from utils import (
    save_as_gif,
    divide_to_macroblocks,
    create_upper_levels,
    motion_exist,
    move_to_low_levels,
    rebuild_image,
)


def process_video(input_file, output_path):
    """
    Perform motion compensation to remove objects in motion from a video.

    Args:
        input_file: Path to the input video file.
        output_path: Path where the output video will be saved.
    """
    logging.info(f"Starting video processing for {input_file}")
    output_file = os.path.join(output_path, f"{Path(input_file).stem}_no_object.avi")
    output_file_gif = os.path.join(
        output_path, f"{Path(input_file).stem}_no_object.gif"
    )
    try:
        vid = cv2.VideoCapture(input_file)

        if not vid.isOpened():
            logging.error(f"Failed to open video file: {input_file}")
            return

        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))

        size = (frame_width, frame_height)
        out = cv2.VideoWriter(
            output_file, cv2.VideoWriter_fourcc(*"MJPG"), fps, size, False
        )

        ret, background = vid.read()
        if not ret:
            logging.error("Failed to read the first frame as background.")
            return

        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        output_frames = []
        with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
            while vid.isOpened():
                ret, curr_frame = vid.read()
                if not ret:
                    break

                curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                hierimg1, hierimg2 = create_upper_levels(background, curr_frame)

                background_macroblocks = divide_to_macroblocks(4, hierimg1[2])
                macroblocked_image = divide_to_macroblocks(4, hierimg2[2])

                movement_blocks = [
                    i
                    for i in range(len(background_macroblocks))
                    if motion_exist(background_macroblocks[i], macroblocked_image[i])
                ]

                background_macroblocks, macroblocked_image, movement_blocks = (
                    move_to_low_levels(movement_blocks, hierimg1, hierimg2)
                )

                for block_idx in movement_blocks:
                    macroblocked_image[block_idx] = background_macroblocks[block_idx]

                corrected_frame = rebuild_image(
                    frame_height // 16, frame_width // 16, np.uint8(macroblocked_image)
                )
                out.write(corrected_frame)
                output_frames.append(corrected_frame)

                background = rebuild_image(
                    frame_height // 16,
                    frame_width // 16,
                    np.uint8(background_macroblocks),
                )
                pbar.update(1)

        vid.release()
        out.release()
        save_as_gif(output_frames, output_file_gif)
        logging.info(f"New video created: {output_file}")
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error occurred during video processing: {str(e)}")


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
        default="OriginalVideos/formula.mp4",
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
        process_video(args.video_path, args.output_path)
        logging.info(f"Video processing completed.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
