import io
import skimage
from torchxrayvision.datasets import normalize


class xrayvision_preproc:
    def __call__(self, key, data):
        with io.BytesIO(data) as stream:
            img = skimage.io.imread(stream)
            img = normalize(img, 255)

            # Check that images are 2D arrays
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                print("error, dimension lower than 2 for image")

            # Add color channel
            img = img[None, :, :]
            return img
