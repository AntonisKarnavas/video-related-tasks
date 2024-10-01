"""
This script converts all video files in a specified directory to GIF format. 
It extracts frames from each video using OpenCV and saves them as GIFs with 
the help of a utility function. The script supports multiple video formats 
and provides logging for tracking the conversion process.
"""

import cv2
import os
import argparse
import logging
from utils import save_as_gif

def extract_frames_from_video(video_path):
    """
    Extracts frames from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        frames (list): List of frames extracted from the video.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR (OpenCV) to RGB (PIL format)

    cap.release()
    return frames

def video_to_gif(video_path, output_gif_path):
    """
    Converts a video to GIF format.

    Args:
        video_path (str): Path to the video file.
        output_gif_path (str): Path to save the GIF file.
    """
    logging.info(f"Starting conversion for {video_path}")
    try:
        frames = extract_frames_from_video(video_path)
        save_as_gif(frames, output_gif_path)
        logging.info(f"Successfully saved GIF to {output_gif_path}")
    except Exception as e:
        logging.error(f"Failed to convert {video_path} to GIF. Error: {str(e)}")

def process_videos_in_directory(directory):
    """
    Processes all video files in the specified directory and converts them to GIFs.

    Args:
        directory (str): Path to the directory containing video files.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # Add more video formats if needed

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(directory, filename)
            gif_filename = os.path.splitext(filename)[0] + '.gif'
            output_gif_path = os.path.join(directory, gif_filename)
            logging.info(f"Processing {video_path} and saving as {output_gif_path}")
            video_to_gif(video_path, output_gif_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all videos in a directory to GIFs.")
    parser.add_argument("directory", type=str, help="Path to the directory containing videos")

    args = parser.parse_args()
    directory = args.directory

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Check if the provided path is a directory
    if not os.path.isdir(directory):
        logging.error(f"The provided path '{directory}' is not a valid directory.")
    else:
        logging.info(f"Processing videos in directory: {directory}")
        process_videos_in_directory(directory)
