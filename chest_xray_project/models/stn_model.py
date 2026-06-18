import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):

    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(True),

            nn.Linear(128, 6)
        )

        self.fc_loc[-1].weight.data.zero_()

        self.fc_loc[-1].bias.data.copy_(
            torch.tensor(
                [1, 0, 0,
                 0, 1, 0],
                dtype=torch.float
            )
        )

        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(16, 1, 3, padding=1)
        )

    def stn(self, x):

        xs = self.localization(x)

        xs = xs.view(xs.size(0), -1)

        theta = self.fc_loc(xs)

        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(
            theta,
            x.size(),
            align_corners=False
        )

        transformed = F.grid_sample(
            x,
            grid,
            align_corners=False,
            padding_mode='border'
        )

        return transformed, theta

    def forward(self, x):

        transformed, theta = self.stn(x)

        refine = self.refine(transformed)

        output = transformed + 0.1 * refine

        return output, theta