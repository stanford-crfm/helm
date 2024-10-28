import base64


def encode_base64(audio_path: str) -> str:
    """Returns the base64 representation of an audio file."""
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
        return base64.b64encode(audio_data).decode("utf-8")
