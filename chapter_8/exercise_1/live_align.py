from utils.capture.realsense import get_camera
import cv2
import numpy as np


def live_align():
    camera = get_camera() 
    aligning = False
    prev_transform = np.eye(3, 3)
    prev_image = None
    prev_descriptors = None
    prev_keypoints = None
    surf = cv2.SIFT_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    while True:
        frames = camera.wait_for_frames()
        
        color_image = np.asanyarray(frames.get_color_frame().get_data())

        if aligning:
            assert prev_image is not None
            assert prev_descriptors is not None
            assert prev_transform is not None

            keypoints, descriptors = surf.detectAndCompute(color_image, None)

            matches = bf.match(prev_descriptors, descriptors)

            if len(matches) > 4:
                src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H_2x3, _ = cv2.estimateAffine2D(dst_pts, src_pts)
                aligned_image = cv2.warpAffine(color_image, H_2x3, (color_image.shape[1], color_image.shape[0]))
                cv2.imshow("Aligned", aligned_image)

        else:
            cv2.imshow("Color", color_image)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

        if cv2.waitKey(50) & 0xFF == ord('a'):
            if not aligning:
                print("Aligning")
                aligning = True
                prev_transform = np.eye(3, 3)
                prev_image = color_image
                prev_keypoints, prev_descriptors = surf.detectAndCompute(color_image, None)
            else:
                print("Not aligning")
                aligning = False
                prev_image = None
                prev_descriptors = None
                prev_keypoints = None

            
if __name__ == "__main__":
    live_align()
