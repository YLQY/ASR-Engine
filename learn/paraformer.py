from funasr import AutoModel
model = AutoModel(model="/home/wenet_data2/tt/asr_server/model/paraformer-zh",device="cuda")
import time
import librosa
t = time.time()
audio, _ = librosa.load("BAC009S0150W0001.wav", sr=16000, mono=True)
print(audio)
print(type(audio))
res = model.generate(input=audio, hotword='')
print(time.time()-t)
print(res)
