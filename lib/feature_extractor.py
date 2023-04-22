import cv2
import numpy as np


def get_features(image)->dict:
    feat_detector = cv2.AKAZE_create()
    kpts, descs = feat_detector.detectAndCompute(image, mask=None)
    features = {"keypoint2d":kpts, "descriptor2d":descs}
    return features