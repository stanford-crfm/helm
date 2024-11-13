from helm.clients.audio_language.llama_omni.model.speech_projector.speech_projector import EncoderProjectorConcat


def build_speech_projector(config):
    projector_type = getattr(config, "speech_projector_type", "linear")
    if projector_type == "linear":
        return EncoderProjectorConcat(config)

    raise ValueError(f"Unknown projector type: {projector_type}")
