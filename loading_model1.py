from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset, Audio

def downloading_model():

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    # ---------------------------------

    dataset = load_dataset(
        "wavs"
    )
    # ---------------------------
    dataset=dataset["train"]
    print(dataset)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(len(dataset))

    return processor,model,dataset

