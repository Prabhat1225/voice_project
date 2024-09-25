

from collections import defaultdict
import matplotlib.pyplot as plt

def speaker(dataset):
    speaker_counts = defaultdict(int)

    for speaker_id in dataset["speaker_id"]:
        speaker_counts[speaker_id] += 1

    plt.figure()
    plt.hist(speaker_counts.values(), bins=20)
    plt.ylabel("Speakers")
    plt.xlabel("Examples")
    plt.show()
    # def select_speaker(speaker_id):
    #     return 100 <= speaker_counts[speaker_id] <= 400

    # dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

    len(set(dataset["speaker_id"]))
    len(dataset)

    return dataset
