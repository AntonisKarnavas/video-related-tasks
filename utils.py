import logging
import numpy as np
import os
from scipy.stats import entropy
from PIL import Image


def save_as_gif(frames, output_gif_path):
    """
    Saves a sequence of frames as a GIF file.

    Args:
        frames (list): List of video frames.
        output_gif_path (str): Path to save the GIF file.
    """
    # Convert frames to a format suitable for PIL
    pil_frames = [Image.fromarray(frame.astype("uint8")) for frame in frames]
    pil_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=40,
        loop=0,
    )


def divide_to_macroblocks(k, array):
    """
    Divide an image array into macroblocks of size k x k.

    Args:
        k (int): Size of the macroblock.
        array (np.ndarray): Input image array.

    Returns:
        np.ndarray: Array of macroblocks.
    """
    macroblocks = []
    for row in range(0, array.shape[0] - k + 1, k):
        for col in range(0, array.shape[1] - k + 1, k):
            macroblocks.append(array[row : row + k, col : col + k].astype("int32"))
    return np.array(macroblocks)


def _hierarchical_division(image):
    """
    Perform a downsampling operation by a factor of 2 on an image.

    Args:
        image (np.ndarray): The image to downsample.

    Returns:
        np.ndarray: The downsampled image.
    """
    return image[::2, ::2]


def create_upper_levels(source, target):
    """
    Subsample the source and target images to create hierarchical levels for motion search.
    Each downsampled image is smaller by a factor of 2.

    Args:
        source (np.ndarray): The source image.
        target (np.ndarray): The target image.

    Returns:
        tuple: Hierarchical image arrays for source and target.
    """
    hierimg1, hierimg2 = [source], [target]

    for _ in range(2):
        source = _hierarchical_division(source)
        target = _hierarchical_division(target)
        hierimg1.append(source)
        hierimg2.append(target)

    return hierimg1, hierimg2


def motion_exist(macroblock1, macroblock2):
    """
    Determine if there is motion between two macroblocks by comparing pixel values.

    Args:
        macroblock1: First macroblock.
        macroblock2: Second macroblock.

    Returns:
        Boolean indicating if motion exists (True/False).
    """
    diff = np.array(macroblock1 - macroblock2)
    num_of_zeros = diff.size - np.count_nonzero(diff)
    return num_of_zeros < 0.9 * diff.size


def move_to_low_levels(movement_blocks, hierimg1, hierimg2):
    """
    Move from the coarsest level (level 3) to finer levels and refine the motion blocks.

    Args:
        movement_blocks (list): List of block indices with detected motion.
        hierimg1 (list): Hierarchical images from the first frame.
        hierimg2 (list): Hierarchical images from the second frame.

    Returns:
        tuple: Refined image macroblocks and motion blocks.
    """
    search = [8, 16]

    for k in range(len(search)):
        image1 = divide_to_macroblocks(search[k], hierimg1[1 - k])
        image2 = divide_to_macroblocks(search[k], hierimg2[1 - k])
        no_movement_blocks = [
            i
            for i in range(len(movement_blocks))
            if not motion_exist(image1[movement_blocks[i]], image2[movement_blocks[i]])
        ]
        movement_blocks = [
            block
            for i, block in enumerate(movement_blocks)
            if i not in no_movement_blocks
        ]

    return image1, image2, movement_blocks


def rebuild_image(height, width, blocks):
    """
    Reconstruct the full image from the array of macroblocks.

    Args:
        height (int): Height of the image.
        width (int): Width of the image.
        blocks (np.ndarray): Array of macroblocks.

    Returns:
        np.ndarray: The reconstructed image.
    """
    rows = []
    for i in range(height):
        row = np.concatenate(blocks[i * width : (i + 1) * width], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def validate_file_path(file_path, description):
    """
    Validates if the given file path exists.

    Args:
        file_path (str): Path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        logging.error(f"{description} file does not exist: {file_path}")
        raise FileNotFoundError(f"{description} file {file_path} not found.")


def entropy_score(frames):
    """
    Calculates the entropy of a sequence of frames.

    Args:
        frames (list): List of video frames.

    Returns:
        float: Entropy of the frames.
    """
    _, counts = np.unique(frames, return_counts=True)
    return entropy(counts)


# Find the neighbor macroblock with the minimum SAD score
def sad(im1, im2, i):
    """
    Compute the Sum of Absolute Differences (SAD) between a macroblock and its neighbors.

    Args:
        im1 (np.ndarray): First image.
        im2 (np.ndarray): Second image.
        i (int): Index of the current macroblock.

    Returns:
        int: Index of the neighbor macroblock with the lowest SAD score.
    """
    block = []
    diff = []
    width = im1.shape[1]

    neighbors = [
        i,
        i + 1,
        i - 1,
        i + width,
        i - width,
        i + width + 1,
        i + width - 1,
        i - width + 1,
        i - width - 1,
    ]
    for n in neighbors:
        if 0 <= n < len(im1):  # Ensure neighbor index is within bounds
            diff.append(_calculate_sad(im2[i], im1[n]))
            block.append(n)

    return block[diff.index(min(diff))]


# Calculate the SAD metric between two macroblocks
def _calculate_sad(macro1, macro2):
    """
    Calculate the Sum of Absolute Differences (SAD) between two macroblocks.

    Args:
        macro1 (np.ndarray): First macroblock.
        macro2 (np.ndarray): Second macroblock.

    Returns:
        int: SAD value.
    """
    return np.sum(np.abs(macro1 - macro2))
