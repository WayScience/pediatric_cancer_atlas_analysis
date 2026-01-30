"""
sweeps.py

Some pre-defined ablation sweeps using Albumentations.
"""

from pathlib import Path
import math

import albumentations as A
import cv2
import numpy as np

from .hooks.albumentations import make_albumentations_hook
from .transforms.dilate import Dilate
from .transforms.erode import Erode


def grid_distort_sweep(
    distort_limit_values=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    num_steps=5,
    variant_prefix="abl_distort",
    **kwargs
):
    """
    Grid distortion sweep

    :param distort_limit_values: List of distort_limit values to sweep over.
    :param num_steps: Number of steps for grid distortion.
    :param variant_prefix: Prefix for variant naming.
    :param kwargs: Additional keyword arguments for the hook.
    :return: Hook function that yields AugVariant for each distortion setting.
    """
    def _hook(src_path: Path):
        
        for distort_limit in distort_limit_values:

            transform = A.GridDistortion(
                distort_limit=distort_limit, # scalar
                num_steps=num_steps, # invariant across sweep
                p=1.0
            )
            g = make_albumentations_hook(
                transform=transform,
                variant_name=f"{variant_prefix}=({distort_limit},{num_steps})",
                **kwargs
            )

            for av in g(src_path):
                av.set_params(
                    fixed=["num_steps"],
                    swept=["distort_limit"],
                    values=[distort_limit]
                )
                yield av

    return _hook


def gauss_noise_sweep(
    std_range_values=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0 ],
    variant_prefix="abl_gaussnoise",
    **kwargs
):
    """
    Gaussian noise sweep

    :param std_range_values: List of standard deviation values to sweep over.
    :param variant_prefix: Prefix for variant naming.
    :param kwargs: Additional keyword arguments for the hook.
    :return: Hook function that yields AugVariant for each noise setting.
    """
    def _hook(src_path: Path):
        
        for std_range in std_range_values:

            transform = A.GaussNoise(
                std_range=(std_range, std_range), # tuple
                p=1.0
            )
            g = make_albumentations_hook(
                transform=transform,
                variant_name=f"{variant_prefix}={std_range}",
                **kwargs
            )

            for av in g(src_path):
                av.set_params(
                    fixed=[],
                    swept=["std_range"],
                    values=[std_range]
                )
                yield av

    return _hook


def blur_sweep(
    its = [10, 20, 30, 40, 50, 60],
    sigma_base = 0.8,
    variant_prefix="abl_blur",
    **kwargs
):
    """
    Gaussian blur sweep
    Approximates repeated blurring by increasing sigma with sqrt(iterations).

    :param its: List of iteration counts to sweep over.
    :param sigma_base: Base sigma value to scale with sqrt(iterations).
    :param variant_prefix: Prefix for variant naming.
    :param kwargs: Additional keyword arguments for the hook.
    :return: Hook function that yields AugVariant for each blur setting.
    """
    def _hook(src_path: Path):
        
        for it in its:

            sigma_limit = sigma_base * math.sqrt(it)

            transform = A.GaussianBlur(
                sigma_limit=(sigma_limit, sigma_limit), # tuple
                p=1.0
            )
            g = make_albumentations_hook(
                transform=transform,
                variant_name=f"{variant_prefix}=({it},{sigma_base})",
                **kwargs
                
            )

            for av in g(src_path):
                av.set_params(
                    fixed=[],
                    swept=["sigma_limit"],
                    values=[sigma_limit]
                )
                yield av

    return _hook


def erode_sweep(
    its = [1, 2, 3, 4, 5, 6],
    k=3,
    **kwargs
):
    """
    Erosion sweep
    Repeatedly applies the erosion operation.

    :param its: List of iteration counts to sweep over.
    :param k: Kernel size for erosion.
    :param kwargs: Additional keyword arguments for the hook.
    :return: Hook function that yields AugVariant for each erosion setting.
    """
    def _hook(src_path: Path):

        for it in its:

            transform = Erode(
                k=k,
                iterations=it,
                always_apply=True,
                p=1.0
            )
            g = make_albumentations_hook(
                transform=transform,
                variant_name=f"abl_erode=({it},{k})",
                **kwargs
            )

            for av in g(src_path):
                av.set_params(
                    fixed=["k"],
                    swept=["iterations"],
                    values=[it]
                )
                yield av

    return _hook


def dilate_sweep(
    its = [1, 2, 3, 4, 5, 6],
    k=3,
    **kwargs
):
    """
    Dilation sweep
    Repeatedly applies the dilation operation.

    :param its: List of iteration counts to sweep over.
    :param k: Kernel size for dilation.
    :param kwargs: Additional keyword arguments for the hook.
    :return: Hook function that yields AugVariant for each dilation setting.
    """
    def _hook(src_path: Path):

        for it in its:

            transform = Dilate(
                k=k,
                iterations=it,
                always_apply=True,
                p=1.0
            )
            g = make_albumentations_hook(
                transform=transform,
                variant_name=f"abl_dilate=({it},{k})",
                **kwargs
            )

            for av in g(src_path):
                av.set_params(
                    fixed=["k"],
                    swept=["iterations"],
                    values=[it]
                )
                yield av

    return _hook


def gamma_sweep(
    # larger gamma for brightening
    # to darken, e.g. use np.geomspace(0.3, 1.0, 6)
    gamma_limit_values=[y * 100 for y in list(np.geomspace(1.0, 3.0, 6))],
    **kwargs
):
    """
    Gamma correction sweep

    :param gamma_limit_values: List of gamma values to sweep over.
    :param kwargs: Additional keyword arguments for the hook.
    :return: Hook function that yields AugVariant for each gamma setting.
    """
    def _hook(src_path: Path):
        
        for gamma_limit in gamma_limit_values:
            
            # prevent floating point issues
            gamma_limit = round(gamma_limit, 2)

            transform = A.RandomGamma(
                gamma_limit=(gamma_limit, gamma_limit), # tuple
                p=1.0
            )
            g = make_albumentations_hook(
                transform=transform,
                variant_name=f"abl_gamma={gamma_limit}",
                **kwargs
            )

            for av in g(src_path):
                av.set_params(
                    fixed=[],
                    swept=["gamma_limit"],
                    values=[gamma_limit]
                )
                yield av

    return _hook


def shift_sweep(
    shift_limit_values=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    scale_limit=(0.0, 0.0),
    rotate_limit=(0, 0),
    border_mode=cv2.BORDER_WRAP,
    variant_prefix="abl_shift",
    **kwargs
):
    """
    Shift sweep

    :param shift_limit_values: List of shift_limit values to sweep over.
    :param scale_limit: Scale limit tuple for ShiftScaleRotate.
    :param rotate_limit: Rotate limit tuple for ShiftScaleRotate.
    :param border_mode: Border mode for ShiftScaleRotate.
    :param variant_prefix: Prefix for variant naming.
    :param kwargs: Additional keyword arguments for the hook.
    :return: Hook function that yields AugVariant for each shift setting.
    """
    def _hook(src_path: Path):
        
        for shift_limit in shift_limit_values:

            transform = A.ShiftScaleRotate(
                shift_limit=(shift_limit, shift_limit), # tuple
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                border_mode=border_mode,
                p=1.0
            )
            g = make_albumentations_hook(
                transform=transform,
                variant_name=f"{variant_prefix}={shift_limit}",
                **kwargs
            )

            for av in g(src_path):
                av.set_params(
                    fixed=["scale_limit", "rotate_limit", "border_mode"],
                    swept=["shift_limit"],
                    values=[shift_limit]
                )
                yield av

    return _hook
