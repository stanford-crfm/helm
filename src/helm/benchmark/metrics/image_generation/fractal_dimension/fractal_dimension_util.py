import numpy as np

from helm.common.optional_dependencies import handle_module_not_found_error


def compute_fractal_dimension(image_path: str) -> float:
    """
    Compute the fractal coefficient of an image.
    From https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension, in fractal
    geometry, the Minkowski–Bouligand dimension, also known as Minkowski dimension
    or box-counting dimension, is a way of determining the fractal dimension of a
    set S in a Euclidean space Rn, or more generally in a metric space (X, d).

    Adapted from https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1.

    :param image_path: Path to the image.
    """

    def fractal_dimension(Z, threshold=0.2):
        # Only for 2d image
        assert len(Z.shape) == 2

        # From https://github.com/rougier/numpy-100 (#87)
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1
            )

            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k * k))[0])

        # Transform Z into a binary array
        Z = Z < threshold

        # Minimal dimension of image
        p = min(Z.shape)

        # Greatest power of 2 less than or equal to p
        n = 2 ** np.floor(np.log(p) / np.log(2))

        # Extract the exponent
        n = int(np.log(n) / np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2 ** np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))

        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    try:
        import cv2
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["heim"])

    image = cv2.imread(image_path, 0) / 255.0  # type: ignore
    assert image.min() >= 0 and image.max() <= 1
    return fractal_dimension(image)
