from transformers import SpeechT5HifiGan
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
from speaker_emm4 import create_speaker_embedding


def prepare_dataset(example,processor):
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


def preapare(processor,dataset,tokenizer):

    main_processor = processor
    processed_example = prepare_dataset(dataset[0],processor)

    list(processed_example.keys())

    tokenizer.decode(processed_example["input_ids"])

    processed_example["speaker_embeddings"].shape

    plt.figure()
    plt.imshow(processed_example["labels"].T)
    plt.show()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    spectrogram = torch.tensor(processed_example["labels"])
    with torch.no_grad():
        speech = vocoder(spectrogram)

    Audio(speech.cpu().numpy(), rate=16000)

    dataset = dataset.map(
        lambda example: prepare_dataset(example, processor),
        remove_columns=dataset.column_names,
    )
    # dataset split train/test---------------
    dataset = dataset.train_test_split(test_size=0.1)
    print(dataset)
    dataset.save_to_disk("voice_project/dataset")
    return dataset,processor



