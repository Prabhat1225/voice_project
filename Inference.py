from IPython.display import Audio
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
import soundfile as sf
import os
import numpy as np
import time
import re
from datasets import load_from_disk
from constant import replacements_hindi

# Create a dictionary for faster lookup
replacement_dict = dict(replacements_hindi)

# Function to transliterate Hindi script to Romanized Hindi
def transliterate_hindi_to_roman(text):
    # Replace each character according to the replacement dictionary
    for hindi_char, roman_char in replacement_dict.items():
        text = text.replace(hindi_char, roman_char)
    return text

# Load the TTS model
model = SpeechT5ForTextToSpeech.from_pretrained("Final_model/checkpoint-464000")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

# Load the vocoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load dataset
# dataset = load_from_disk("output_folder/dataset")
# Access the 304th item in the test set
# example = dataset["test"][304]
# speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

speaker_embeddings = torch.tensor(np.load("/home/prabhat.singh/Project/speakerembedding.npy"))

# Input text in Hindi script
text="""भारत एक विविधता से भरा हुआ देश है जो दक्षिण एशिया में स्थित है। यह दुनिया का दूसरा सबसे अधिक जनसंख्या वाला देश है और यहाँ की सांस्कृतिक, भाषाई, और भौगोलिक विविधता इसे एक अनोखा देश बनाती है। भारत की राजधानी नई दिल्ली है और यह अट्ठाईस राज्यों और आठ केंद्र शासित प्रदेशों में बाँटा गया है।

भारत की आधिकारिक भाषाएँ हिंदी और अंग्रेज़ी हैं, लेकिन यहाँ बाईस से अधिक भाषाएँ और सैकड़ों बोलियाँ बोली जाती हैं। यहाँ की प्रमुख धर्मों में हिंदू धर्म, इस्लाम, ईसाई धर्म, सिख धर्म, बौद्ध धर्म, और जैन धर्म शामिल हैं।

भारत का इतिहास बहुत पुराना और समृद्ध है, जिसमें प्राचीन सभ्यताओं, महान साम्राज्यों, और विविध सांस्कृतिक धरोहरों का योगदान है। यहाँ की कला, संगीत, नृत्य, और भोजन भी विश्व प्रसिद्ध हैं।"""
# Split text based on punctuation marks
sentences = re.split(r'(?<=[.,।])\s+', text)  # Split at punctuation followed by whitespace

# Start processing
all_speech_chunks = []

for idx, sentence in enumerate(sentences):
    start_time = time.time()
    
    # Transliterate to Romanized Hindi
    romanized_sentence = transliterate_hindi_to_roman(sentence)
    
    print(f"Processing sentence {idx+1}: {romanized_sentence}")
    
    # Convert sentence to input tensor
    inputs = processor(text=romanized_sentence, return_tensors="pt")

    # Generate spectrogram for the sentence
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    # Use the vocoder to convert the spectrogram to speech
    with torch.no_grad():
        speech = vocoder(spectrogram)

    # Convert speech to NumPy array
    speech_numpy = speech.cpu().numpy().flatten()  # Ensure it is 1D

    # Accumulate the speech chunks
    all_speech_chunks.append(speech_numpy)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time for sentence {idx+1}: {inference_time:.2f} seconds")

# Concatenate all speech chunks into one array
full_audio = np.concatenate(all_speech_chunks)

# Save the concatenated audio to a .wav file
output_dir = "output_folder/Hello_wavs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

final_output_path = os.path.join(output_dir, "full_text_output_hindi.wav")
sf.write(final_output_path, full_audio, samplerate=16000)
print(f"Full audio saved at: {final_output_path}")

# Play the full generated audio
Audio(full_audio, rate=16000)

print("Full text processed and saved as a single .wav file.")
