# file_1.py
import subprocess

print("Running speaker3.py")
# Call the next file
subprocess.run(["python", "speaker_emm4.py"])




from collections import defaultdict
speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1

import matplotlib.pyplot as plt

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
