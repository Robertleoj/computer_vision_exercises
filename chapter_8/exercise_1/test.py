# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

from pathlib import Path
import mediapy
import cv2

# +
vid_path = Path("../../data/eva_cutie/recording_00000/rgb.mp4")

frames = mediapy.read_video(vid_path)

mediapy.show_video(frames)
# -

# detect features with SURF
surf = cv2.ORB_create(nfeatures=50)

# +
im1 = frames[0]
im2 = frames[1]

kp1, des1 = surf.detectAndCompute(im1, None)
kp2, des2 = surf.detectAndCompute(im2, None)

# +
# plot on the images
im1_kp = cv2.drawKeypoints(im1, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im2_kp = cv2.drawKeypoints(im2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

mediapy.show_images([im1_kp, im2_kp])

# +
images_to_match = frames[:]

features_per_img = 5000
surf = cv2.ORB_create(nfeatures=features_per_img)

img_keypoints = []
keypoint_descriptors = []

for img in images_to_match:
    kp, des = surf.detectAndCompute(img, None)
    img_keypoints.append(kp)
    keypoint_descriptors.append(des)

# +
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# match descriptors
matches = bf.match(keypoint_descriptors[0], keypoint_descriptors[1])

# sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

mediapy.show_images([images_to_match[0], images_to_match[1]])
mediapy.show_image(cv2.drawMatches(images_to_match[0], img_keypoints[0], images_to_match[1], img_keypoints[1], matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))


# +
import numpy as np

# match images in the entire video
aligned_images = []
transforms = [np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])]
aligned_images.append(images_to_match[0])

for i in range(1, len(images_to_match)):
    img1 = images_to_match[i - 1]
    img2 = images_to_match[i]

    kp1 = img_keypoints[i - 1]
    kp2 = img_keypoints[i]

    des1 = keypoint_descriptors[i - 1]
    des2 = keypoint_descriptors[i]

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:10]

    matches_im1 = np.array([kp1[m.queryIdx].pt for m in matches])
    matches_im2 = np.array([kp2[m.trainIdx].pt for m in matches])

    # compute rotation and translation
    H_2x3, t = cv2.estimateAffine2D(matches_im2, matches_im1)
    H_3x3 = np.vstack([H_2x3, [0, 0, 1]])

    prev_transform = transforms[i - 1]
    new_transform = prev_transform @ H_3x3

    transforms.append(new_transform)

    # align the second image to the first one
    aligned_img = cv2.warpAffine(img2, new_transform[:2, :], (img1.shape[1], img1.shape[0]))

    aligned_images.append(aligned_img)

mediapy.show_video(aligned_images, fps=30, downsample=False, width=1500)


