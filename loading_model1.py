# file_1.py
import subprocess

print("Running loading_model1.py")
# Call the next file
subprocess.run(["python", "clean_up2.py"])



from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# ---------------------------------
from datasets import load_dataset, Audio

dataset = load_dataset(
    "/content/drive/MyDrive/speecht5/dataset/data/wavs"
)
# ---------------------------
dataset=dataset["train"]
print(dataset)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
len(dataset)
