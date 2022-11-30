from collections import namedtuple
from itertools import accumulate
from typing import Optional

import cv2
import numpy as np
import torch

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> torch.Tensor:
    '''Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    '''
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0])
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype='uint8')
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index
    return torch.FloatTensor(colorwheel)


WHEEL = _make_colorwheel()
N_COLS = len(WHEEL)
WHEEL = torch.vstack((WHEEL, WHEEL[0]))  # Make the wheel cyclic for interpolation

# Adapted from https://github.com/facebookresearch/banmo/blob/main/third_party/ext_utils/flowlib.py

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def cat_imgflo(img: torch.Tensor, flow: torch.Tensor, skip: int = None) -> torch.Tensor:
    """
    img in (0,1)
    flo in non-normalized coordinate
    """
    flow = flow.clone()
    flow[:, :, 0] /= flow.shape[1]
    flow[:, :, 1] /= flow.shape[0]

    img = img.clone() * 255
    h, w = img.shape[:2]
    flow = flow.clone()
    flow[:, :, 0] = flow[:, :, 0] * 0.5 * w
    flow[:, :, 1] = flow[:, :, 1] * 0.5 * h
    imgflo = _point_vec(img, flow, skip)
    return imgflo


def _point_vec(img: torch.Tensor, flow: torch.Tensor, skip: int = None) -> torch.Tensor:
    if skip is None:
        skip = min(10, 10 * img.shape[1] // 500)

    dispimg = img.clone().cpu().numpy()
    meshgrid = np.meshgrid(range(dispimg.shape[1]), range(dispimg.shape[0]))

    colorflow = _flow_to_image(flow.clone().cpu()).int()
    for i in range(dispimg.shape[1]):  # x
        for j in range(dispimg.shape[0]):  # y
            if flow.shape[-1] == 3 and flow[j, i, 2] != 1: continue
            if j % skip != 0 or i % skip != 0: continue
            leng = torch.linalg.norm(flow[j, i, :2]).item()
            if leng < 1:
                continue
            xend = int((meshgrid[0][j, i] + flow[j, i, 0]))
            yend = int((meshgrid[1][j, i] + flow[j, i, 1]))
            dispimg = cv2.arrowedLine(dispimg, (meshgrid[0][j, i], meshgrid[1][j, i]), \
                                      (xend, yend),
                                      (int(colorflow[j, i, 2]), int(colorflow[j, i, 1]), int(colorflow[j, i, 0])), 1,
                                      tipLength=4 / leng, line_type=cv2.LINE_AA)
    return torch.FloatTensor(dispimg).to(img.device) / 255.


def _flow_to_image(flow: torch.Tensor) -> torch.Tensor:
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idxUnknow = (u.abs() > UNKNOWN_FLOW_THRESH) | (v.abs() > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    rad = torch.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, rad.max())

    u = u / (maxrad + 1e-8)
    v = v / (maxrad + 1e-8)

    img = _compute_color(u, v)

    idx = idxUnknow.unsqueeze(-1).expand(-1, -1, 3)
    img[idx] = 0

    return img.byte()


def _compute_color(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = torch.zeros(h, w, 3, device=u.device)
    nanIdx = torch.isnan(u) | torch.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    rad = torch.sqrt(u ** 2 + v ** 2)

    a = torch.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (N_COLS - 1) + 1

    k0 = torch.floor(fk).long()

    k1 = k0 + 1
    k1[k1 == N_COLS + 1] = 1
    f = fk - k0

    for i in range(0, WHEEL.shape[1]):
        tmp = WHEEL[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = torch.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = (torch.floor(255 * col * (1 - nanIdx.int()))).byte()

    return img


# Adapted from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/visualize.py
def scene_flow_to_rgb(
        flow: torch.Tensor,
        flow_max_radius: Optional[float] = None,
        background: Optional[str] = 'dark',
) -> torch.Tensor:
    '''Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    '''
    valid_backgrounds = ('bright', 'dark')
    if background not in valid_backgrounds:
        raise ValueError(f'background should be one the following: {valid_backgrounds}, not {background}.')

    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = torch.abs(complex_flow), torch.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = torch.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((N_COLS - 1) / (2 * np.pi))

    # Interpolate the hues
    angle_fractional, angle_floor, angle_ceil = torch.fmod(angle, 1), angle.trunc(), torch.ceil(angle)
    angle_fractional = angle_fractional.unsqueeze(-1)
    wheel = WHEEL.to(angle_floor.device)
    float_hue = (
            wheel[angle_floor.long()] * (1 - angle_fractional) + wheel[angle_ceil.long()] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        'ColorizationArgs', ['move_hue_valid_radius', 'move_hue_oversized_radius', 'invalid_color']
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * factors.unsqueeze(-1)

    def move_hue_on_S_axis(hues, factors):
        return 255. - factors.unsqueeze(-1) * (255. - hues)

    if background == 'dark':
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, torch.FloatTensor([255, 255, 255])
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, torch.zeros(3))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    return colors / 255.


def label_colormap(N: int) -> torch.FloatTensor:
    cmap = torch.zeros(N, 3)
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (_bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (_bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (_bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    return cmap / 255.


def _bitget(byteval: int, idx: int) -> int:
    return ((byteval & (1 << idx)) != 0)
