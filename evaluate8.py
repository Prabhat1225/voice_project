# file_10.py
print("Running evaluate8.py")



model = SpeechT5ForTextToSpeech.from_pretrained("/content/drive/MyDrive/speecht5/model/checkpoint-8000")

example = dataset["test"][304]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
speaker_embeddings.shape

text = "अपनी ओर से पूरी कोशिश की, नतीजों को लेकर घबराहट नहीं|"

inputs = processor(text=text, return_tensors="pt")

spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

plt.figure()
plt.imshow(spectrogram.T)
plt.show()

with torch.no_grad():
    speech = vocoder(spectrogram)

from IPython.display import Audio
Audio(speech.numpy(), rate=16000)
import soundfile as sf
sf.write("output.wav", speech.numpy(), samplerate=16000)
