from typing import Sequence, Optional, Tuple

import numpy as np

def generate_random_crops(
    image: np.ndarray,
    crop_size: int,
    num_crops: int,
    random_state: Optional[np.random.RandomState] = None,
    **kwargs: np.ndarray 
) -> Tuple[np.ndarray, ...]:
    """
    Generate random crops from the input image.

    :param image: Input image as a numpy array of shape (C, H, W).
    :param crop_size: Size of the square crop (crop_size x crop_size).
    :param num_crops: Number of random crops to generate.
    :param random_state: Optional numpy RandomState for reproducibility.
    :param kwargs: keyword arguments to specify additional payload arrays to crop in the same way.
    
    """
    if random_state is None:
        random_state = np.random.RandomState()

    c, h, w = image.shape
    if crop_size > h or crop_size > w:
        raise ValueError("Crop size must be smaller than image dimensions.")
    
    payloads = [image] + [
        kwargs[i] for i in kwargs if isinstance(kwargs[i], np.ndarray)
    ]    
    crops = [
        np.empty((num_crops, p.shape[0], crop_size, crop_size), dtype=p.dtype) for p in payloads
    ]

    for i in range(num_crops):
        top = random_state.randint(0, h - crop_size + 1)
        left = random_state.randint(0, w - crop_size + 1)

        for j, p in enumerate(payloads):
            crops[j][i] = p[:, top:top + crop_size, left:left + crop_size]

    return tuple(crops)
