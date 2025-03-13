import ctranslate2
import librosa
import transformers
import time
# Load and resample the audio file.
audio, _ = librosa.load("BAC009S0150W0001.wav", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# Load the model on CPU.
model = ctranslate2.models.Whisper("/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo-ct2",device="cuda")

# Detect the language.
results = model.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))

# Describe the task in the prompt.
# See the prompt format in https://github.com/openai/whisper.
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)
print(prompt)
t = time.time()
# Run generation for the 30-second window.
results = model.generate(features, [prompt])
transcription = processor.decode(results[0].sequences_ids[0])
print(time.time()-t)
print(transcription)
