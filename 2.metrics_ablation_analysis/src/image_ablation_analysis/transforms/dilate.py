"""
dilate.py

Dilate transform for image ablation analysis.
"""

import cv2
import numpy as np
import albumentations as A


class Dilate(A.ImageOnlyTransform):
    """
    Dilate transform using OpenCV to evenly dilate brighter regions.

    :param k: Kernel size (will be made odd if even).
    :param iterations: Number of dilation iterations.
    :param always_apply: If True, always apply the transform.
    :param p: Probability of applying the transform.
    """
    def __init__(self, k: int = 3, iterations: int = 1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.k, self.iterations = k | 1, iterations
    def apply(self, img, **params):
        kernel = np.ones((self.k, self.k), np.uint8)
        return cv2.dilate(img, kernel, iterations=self.iterations)
    def get_transform_init_args_names(self):
        return ("k", "iterations")
    def __repr__(self):
        return f"Dilate(k={self.k}, iterations={self.iterations})"
