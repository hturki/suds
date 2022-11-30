import torch
from scipy.spatial import Delaunay


# Modified from https://github.com/pytorch/pytorch/issues/50339#issuecomment-1339910414
class GridInterpolator:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

        self.Y_grid, self.X_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        self.X_grid = self.X_grid.reshape(-1)
        self.Y_grid = self.Y_grid.reshape(-1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Tesselate grid points
        pos = torch.stack([X, Y], dim=-1).cpu().numpy()
        tri = Delaunay(pos, furthest_site=False)

        # Find the corners of each simplice
        corners_X = X[tri.simplices]
        corners_Y = Y[tri.simplices]
        corners_F = value[tri.simplices]

        # Find simplice ID for each query pixel in the original grid
        pos_orig = torch.stack([self.X_grid, self.Y_grid], dim=-1).numpy()
        simplice_id = tri.find_simplex(pos_orig)

        # Find X,Y,F values of the 3 nearest grid points for each
        # pixel in the original grid
        corners_X_pq = corners_X[simplice_id]
        corners_Y_pq = corners_Y[simplice_id]
        corners_F_pq = corners_F[simplice_id]

        x1, y1 = corners_X_pq[:, 0], corners_Y_pq[:, 0]
        x2, y2 = corners_X_pq[:, 1], corners_Y_pq[:, 1]
        x3, y3 = corners_X_pq[:, 2], corners_Y_pq[:, 2]

        x_grid_gpu = self.X_grid.to(X)
        y_grid_gpu = self.Y_grid.to(X)
        lambda1 = ((y2 - y3) * (x_grid_gpu - x3) + (x3 - x2) * (y_grid_gpu - y3)) / \
                  ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))

        lambda2 = ((y3 - y1) * (x_grid_gpu - x3) + (x1 - x3) * (y_grid_gpu - y3)) / \
                  ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))

        lambda3 = 1 - lambda1 - lambda2

        out = lambda1 * corners_F_pq[:, 0] + lambda2 * corners_F_pq[:, 1] + lambda3 * corners_F_pq[:, 2]
        out[simplice_id == -1] = 0

        return out.reshape(self.height, self.width)
