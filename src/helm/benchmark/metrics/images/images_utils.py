import cv2


def is_blacked_out_image(image_path: str) -> bool:
    """Returns True if the image is all black. False otherwise."""
    image = cv2.imread(image_path, 0)
    return cv2.countNonZero(image) == 0
