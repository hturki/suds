from typing import Tuple

import torch

from suds.data.image_metadata import ImageMetadata


def get_w2c_and_K(item: ImageMetadata) -> Tuple[torch.Tensor, torch.Tensor]:
    K = torch.eye(3)
    K[0, 0] = item.intrinsics[0]
    K[1, 1] = item.intrinsics[1]
    K[0, 2] = item.intrinsics[2]
    K[1, 2] = item.intrinsics[3]

    c2w_4x4 = torch.eye(4)
    c2w_4x4[:3] = item.c2w
    w2c = torch.inverse(c2w_4x4)

    return w2c, K
