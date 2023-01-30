"""NAIP Geographic dataset."""

import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset


class NAIPGeo(NonGeoDataset):
    """NAIPGeo dataset.

    This dataset contains 100,000 256x256 patches of NAIP imagery sampled uniformly at
    random from the Microsoft Planetary Computer.
    """

    validation_filenames = [
        "index.csv",
        "images/",
        "images/sample_0.tif",
        "images/sample_99999.tif",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new NAIPGeo dataset instance.

        Args:
            root: root directory of NAIP pre-sampled dataset
            transform: torch transform to apply to a sample
        """
        self.root = root
        self.transform = transform

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        df = pd.read_csv(os.path.join(self.root, "index.csv"))
        self.filenames = []
        self.points = []
        for i in range(df.shape[0]):
            self.filenames.append(
                os.path.join(self.root, "images", df.iloc[i]["filename"])
            )
            self.points.append((df.iloc[i]["lon"], df.iloc[i]["lat"]))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        with rasterio.open(self.filenames[index]) as f:
            data = f.read()

        img = torch.tensor(data)
        point = torch.tensor(self.points[index])

        sample = {"image": img, "point": point}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.filenames)

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        ncols = 1

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        ax.imshow(image[:, :, :3])
        ax.axis("off")

        if show_titles:
            ax.set_title(f"({sample['point'][0]:0.4f}, {sample['point'][1]:0.4f})")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
