import logging
import pyrealsense2 as pyrs
import cv2
import numpy as np
import tyro
from pathlib import Path
import mediapy

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def save_recording(output_folder: Path, rgb_frames: list[np.ndarray], depth_frames: list[np.ndarray]) -> None:
    output_folder.mkdir(exist_ok=True, parents=True)
    mediapy.write_video(output_folder / "rgb.mp4", rgb_frames)
    mediapy.write_video(output_folder / "depth.mp4", depth_frames)

def main(output_folder: Path, framerate: int = 30) -> None:
    # Create a pipeline
    pipeline = pyrs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = pyrs.config()
    config.enable_stream(pyrs.stream.depth, 640, 480, pyrs.format.z16, framerate)
    config.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.bgr8, framerate)

    # Start streaming
    pipeline.start(config)

    recording = False
    recording_index = 0
    color_frames = []
    depth_frames = []
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            if recording:
                color_frames.append(color_image.copy())
                depth_frames.append(depth_image.copy())

            # Show images
            cv2.imshow('RealSense', images)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            if cv2.waitKey(10) & 0xFF == ord('r'):
                if recording:
                    logger.info(f"Saving recording {recording_index}")
                    save_recording(output_folder / f"recording_{recording_index:05d}", color_frames, depth_frames)
                    recording_index += 1
                    recording = False
                    color_frames = []
                    depth_frames = []
                else:
                    logger.info(f"Starting recording {recording_index}")
                    recording = True

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tyro.cli(main)
