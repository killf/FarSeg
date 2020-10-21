from PIL import Image
from collections.abc import Sequence, Iterable

from torchvision.transforms.functional import _is_pil_image


def resize(img, label, size, interpolation=Image.BILINEAR):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img, label
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation), label.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation), label.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation), label.resize(size[::-1], interpolation)

