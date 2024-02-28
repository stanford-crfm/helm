from helm.common.media_object import MediaObject, MultimediaObject


def get_single_image_multimedia_object(image_location: str) -> MultimediaObject:
    """
    Returns a `MultimediaObject` containing a single image file used for text-to-image generation clients.
    """
    file_extension: str = image_location.split(".")[-1]
    return MultimediaObject([MediaObject(content_type=f"image/{file_extension}", location=image_location)])
