from helm.clients.audio_language.llama_omni.model.speech_encoder.speech_encoder import WhisperWrappedEncoder


def build_speech_encoder(config):
    speech_encoder_type = getattr(config, "speech_encoder_type", "none")
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)

    raise ValueError(f"Unknown speech encoder: {speech_encoder_type}")
