# file_1.py
import subprocess

print("Running prepare_dataset5.py")
# Call the next file
subprocess.run(["python", "collator6.py"])




def prepare_dataset(example):
    # load the audio data; if necessary, this resamples the audio to 16kHz
    audio = example["audio"]

    # feature extraction and tokenization
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example


processed_example = prepare_dataset(dataset[0])

list(processed_example.keys())

tokenizer.decode(processed_example["input_ids"])

processed_example["speaker_embeddings"].shape

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(processed_example["labels"].T)
plt.show()

from transformers import SpeechT5HifiGan
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

spectrogram = torch.tensor(processed_example["labels"])
with torch.no_grad():
    speech = vocoder(spectrogram)

from IPython.display import Audio
Audio(speech.cpu().numpy(), rate=16000)

dataset = dataset.map(
    prepare_dataset, remove_columns=dataset.column_names,
)
# dataset split train/test---------------
dataset = dataset.train_test_split(test_size=0.1)
dataset
