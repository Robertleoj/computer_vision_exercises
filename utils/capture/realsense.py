import pyrealsense2 as pyrs

def get_camera() -> pyrs.pipeline:
    pipeline = pyrs.pipeline()
    config = pyrs.config()
    config.enable_stream(pyrs.stream.depth, 640, 480, pyrs.format.z16, 30)
    config.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.bgr8, 30)
    pipeline.start(config)

    return pipeline
