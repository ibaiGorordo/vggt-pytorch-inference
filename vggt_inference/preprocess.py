import cv2
import numpy as np
import torch

def load_images(image_paths, downsample_factor=1):
    """
    Load images from the specified paths using OpenCV.

    Args:
        image_paths (list): List of image file paths.
        downsample_factor (int, optional): Factor by which to downsample the images. Default is 1 (no downsampling).

    Returns:
        list: List of loaded images as numpy arrays.
    """
    images = []
    for path in image_paths[::downsample_factor]:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image at {path}")
        images.append(img)
    return images

def read_images_from_video(video_path, downsample_factor=1):
    """
    Read images from a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.
        downsample_factor (int, optional): Factor by which to downsample the images. Default is 1 (no downsampling).

    Returns:
        list: List of loaded images as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    images = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % downsample_factor == 0:
            images.append(frame)
        frame_count += 1

    cap.release()
    return images

def preprocess_images(images, mode="crop"):
    """
    Preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        images (list): List of images loaded using OpenCV (numpy arrays in BGR format).
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid
    """
    if len(images) == 0:
        raise ValueError("At least 1 image is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    processed_images = []
    shapes = set()
    target_size = 518

    for img in images:
        # Convert BGR (OpenCV format) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize image
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # Convert to tensor and normalize to (0, 1)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y:start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        processed_images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img in processed_images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        processed_images = padded_images

    processed_images = torch.stack(processed_images)

    if len(images) == 1 and processed_images.dim() == 3:
        processed_images = processed_images.unsqueeze(0)

    return processed_images