import rerun as rr

from vggt_inference import VGGTInference
from vggt_inference.visualize import visualize_results
from vggt_inference.preprocess import read_images_from_video


# Initialize the VGGTInference class
vggt_inference = VGGTInference()

# Downsample factor, increase to reduce the number of images if your GPU is running out of memory
downsample_factor = 1

# Load and preprocess example images (replace with your own image paths)
images = read_images_from_video("assets/video.mp4", downsample_factor=downsample_factor)
print("Number of images:", len(images))

# Run inference
results = vggt_inference(images)

# Visualize the results using Rerun
rr.init("vggt_inference", spawn=True)

visualize_results(results, filter_percent=40)



