from PIL import Image
from nerfstudio.data.datasets.base_dataset import InputDataset

import numpy as np
import numpy.typing as npt

from suds.stream_utils import image_from_stream


class StreamInputDataset(InputDataset):

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = str(self._dataparser_outputs.image_filenames[image_idx])
        # Converting to path mangles s3:// -> s3:/
        pil_image = image_from_stream(image_filename.replace('s3:/', 's3://').replace('gs:/', 'gs://'))
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is incorrect."
        return image