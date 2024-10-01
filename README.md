# Video Compression and Object Removal Tasks

## Video Compression & Decompression using Error Frames

### Overview

This script encodes and decodes error frames from a given video sequence without motion compensation. It assumes the first frame is an I-frame, and the subsequent frames are predicted (P-frames) by calculating the difference between consecutive frames. The error images are then saved and encoded in binary format. The decoder reconstructs the video using the encoded error frames, and the script also computes and compares the entropy of the original grayscale frames and the error frames to show differences in information content.

### Background Knowledge

#### Video Compression Concepts

When working with video compression, we often deal with different types of frames:

1. **I-frames (Intra-coded frames)**: These are keyframes that store a complete image of the scene. They are not dependent on other frames for decoding.
2. **P-frames (Predictive frames)**: Theseframes store only the changes (differences or "errors") between the current frame and the previous frame. P-frames help reduce data redundancy by not storing the entire frame.
3. **Error Frames**: These represent the pixel-wise difference between a P-frame and the preceding frame (whether it's an I-frame or another P-frame). The goal is to capture only the "error" or change, which results in smaller storage requirements.

#### Entropy in Image Processing

**Entropy** is a measure of the amount of information or uncertainty in data. In image processing:

- Higher entropy indicates more detailed and complex images, with more variation between pixels.
- Lower entropy suggests less variation and simpler, more uniform images.

By calculating the entropy of original grayscale video frames and comparing it to the entropy of the error frames, we can quantify the reduction in information. The lower entropy of the error frames reflects the fact that they contain only the changes between consecutive frames, which is a subset of the original frame's information.

#### Encoding and Decoding

In this script:

- The **encoder** calculates the difference between consecutive frames (P-frames) and saves them along with the first I-frame. This sequence is stored in a binary format.
- The **decoder** reconstructs the video by adding the error frames back to the previous frame, recreating the original sequence.

### Key Observations

- **Entropy Comparison**: The original video, in grayscale, has a higher entropy compared to the sequence of error frames. The error images contain less information since they represent only the changes between frames. For example, in areas like the sky, where there is little change between frames, the error frames are very dark with minimal detail.

![error frames gif](video_compression_decompression_using_error_frames/error_frames.gif)
  
- **Quality Degradation in Decoding**: The reconstructed video, generated from the error frames, may exhibit lower quality, especially in areas with little change, such as clouds or sky, where some pixelation may occur.

![recostructed image gif](video_compression_decompression_using_error_frames/reconstructed_video.gif)

### Usage

To execute the script and utilize its functionality, you can run the following command. Make sure to specify the path to the input video and the desired output directory for the results. By default, the script uses `OriginalVideos/lion.mp4` as the input video.

```bash
python video_compression_decompression_using_error_frames.py --video_path "path_to_input_video.mp4" --output_path "path_to_output_directory"
```

#### Parameters

* --video_path: Path to the input video file (default: OriginalVideos/lion.mp4).
- --output_path: Directory where the output files will be saved.

---

## Video Compression & Decompression using Motion Compensation

### Overview

This project implements motion compensation as a video compression technique. Motion compensation is commonly used in video compression algorithms to reduce redundancy between frames. This project focuses on breaking down video frames into macroblocks, comparing them using the Sum of Absolute Differences (SAD) metric, and compressing the frames based on the detected motion. It also includes entropy analysis to evaluate the compression efficiency and a visualization of predicted and error images.

### Background Knowledge

#### Video Compression with Motion Compensation

In video, consecutive frames often contain similar regions. Motion compensation is a method that leverages this similarity by encoding the motion of macroblocks (small rectangular blocks of pixels) between frames. Instead of storing entire frames, we store the first frame and the motion vectors that describe how blocks move from one frame to the next. These motion vectors are used by the decoder to reconstruct the subsequent frames.

#### Macroblock and Motion Estimation

Each frame is divided into macroblocks, typically of size 16x16 pixels. The goal is to find the motion vector for each macroblock, representing the displacement between the current and reference frames. The project uses a search radius of 16 pixels and compares macroblocks using the Sum of Absolute Differences (SAD) metric.

#### Hierarchical Search

To optimize the motion search, we can implement a hierarchical search that performs the motion search on downsampled versions of the image. This reduces the computational load without sacrificing too much accuracy.

#### Key Concepts

1. **Sum of Absolute Differences (SAD):**
   SAD is a simple and efficient method to measure the similarity between two macroblocks. It sums the absolute differences between corresponding pixel values in the two blocks.

2. **Entropy:**
   Entropy measures the amount of information in an image. In video compression, lower entropy values indicate that the image has less information, making it easier to compress. We calculate and compare the entropy of the original video frames with the error images generated during motion compensation.

3. **Motion Compensation:**
   This technique calculates motion vectors by comparing macroblocks of the current frame with the previous frame, reducing the need to store large amounts of data for similar frames.

### Workflow

1. **Encoding Process:**
   - Divide the video into frames and convert them to grayscale.
   - Break each frame into 16x16 pixel macroblocks.
   - Calculate motion vectors using SAD with a search radius of 16 pixels.
   - Apply hierarchical search to downsample the frames and improve efficiency.
   - Generate predicted frames and error frames (the difference between the original and predicted frames).

2. **Decoding Process:**
   - Use the motion vectors to reconstruct the frames from the reference frame.
   - Combine the error frames and the predicted frames to regenerate the video sequence.

### Results

- **Entropy Comparison:** The entropy of the original grayscale video was higher than the entropy of the error frames. This indicates that the error frames contain less information and are easier to compress.
- **Image Quality:** Although motion compensation reduces data, there is a noticeable loss in quality in regions with complex textures, such as grass. However, areas like the sky, where neighboring blocks are similar, show less error.
![decoded frames gif](video_compression_decompression_using_motion_compensation/decoded_frames.gif)

### Usage

To execute the script and utilize its functionality, you can run the following command. Make sure to specify the path to the input video and the desired output directory for the results. By default, the script uses `OriginalVideos/lion.mp4` as the input video.

```bash
python video_compression_decompression_using_motion_compensation.py --video_path "path_to_input_video.mp4" --output_path "path_to_output_directory"
```

#### Parameters

* --video_path: Path to the input video file (default: OriginalVideos/lion.mp4).
- --output_path: Directory where the output files will be saved.

---

## Video Object Removal using Motion Compensation

### Overview

This task demonstrates how to remove an object from a video using motion compensation techniques. The task involves taking a short video clip, identifying a moving object, and creating a new version of the video where that object is no longer present. This process utilizes a method known as **macroblock-based motion compensation**, which is widely employed in video compression and reconstruction techniques.

The core of the process is identifying the motion between frames and replacing moving blocks with background blocks from a reference frame, thus "removing" the object.

### Background Knowledge

The necessary background knowledge for understanding this project has already been explained in the sections above. The key concept is motion compensation, where macroblocks from a reference frame (usually the first frame) are used to replace blocks in subsequent frames that show movement. This method is often used in video compression formats such as MPEG and H.264 to reduce file size and optimize playback efficiency.

### Task breakdown

#### Objective

The task is to remove an object from a video that shows mild movement (of objects and the camera). The object will disappear throughout the video by replacing the blocks in which it appears with static blocks from a background frame.

#### Approach

1. Frame Selection: The first frame of the video is treated as the reference background.
2. Motion Detection: For each subsequent frame, the algorithm detects movement by analyzing changes in macroblocks.
3. Motion Compensation: For each moving block in the target frame, the corresponding block from the reference background is substituted.
4. Object Removal: Since the background remains static and only the moving macroblocks are replaced, the object in motion will be "removed" from the video.

#### Object Selection

If the video contains multiple moving objects, the object to be removed is selected by editing the first frame to eliminate the object. The algorithm will then propagate this change to the rest of the video.

### Results

| Before object removal                       | After object removal                        |
|----------------------------------|----------------------------------|
| ![formula before](OriginalVideos/formula.gif) | ![formula after](object_removal/formula_no_object.gif) |
| ![ball before](OriginalVideos/ball.gif) | ![ball after](object_removal/ball_no_object.gif) |
| ![owl before](OriginalVideos/owl.gif) | ![owl after](object_removal/owl_no_object.gif) |

### Usage

To execute the script and utilize its functionality, you can run the following command. Make sure to specify the path to the input video and the desired output directory for the results. By default, the script uses `OriginalVideos/formula.mp4` as the input video. You can also use the following videos: `OriginalVideos/ball.mp4` and `OriginalVideos/owl.mp4`

```bash
python object_removal.py --video_path "path_to_input_video.mp4" --output_path "path_to_output_directory"
```

#### Parameters

* --video_path: Path to the input video file (default: OriginalVideos/formula.mp4).
- --output_path: Directory where the output files will be saved.

---

## Extra scripts

### Video to GIF Converter

This script converts all video files in a specified directory to GIF format.

### Usage

To run the script, use the following command:

```bash
python get_gif_version_of_video.py /path/to/video/directory
```

Replace `/path/to/video/directory` with the path to the directory containing your video files.

Here's a completed section for your README under "Getting Started":

---

## Getting Started

### Prerequisites

- Python 3.x
- Virtualenv for creating isolated Python environments

### Installation and Running

1. **Clone the repository** (if applicable) or download the script.

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   virtualenv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required libraries from requirements.txt file**:

   ```bash
   pip install -r requirements.txt
   ```
