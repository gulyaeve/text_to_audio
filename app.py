from transformers import BarkModel, AutoProcessor
import torch
import scipy


def text_to_audio(bark_model="suno/bark", voice_preset="v2/ru_speaker_6", text="some text"):
    model = BarkModel.from_pretrained(bark_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(bark_model)

    inputs = processor(text, voice_preset=voice_preset).to(device)
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(f"{voice_preset.split('/')[0]}", rate=sample_rate, data=audio_array)


if __name__ == '__main__':
    text_to_audio()

