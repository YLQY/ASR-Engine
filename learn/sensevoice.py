from funasr import AutoModel
model = AutoModel(model="/home/wenet_data2/tt/asr_server/model/sensevoice",device="cuda")
import time
t = time.time()
res = model.generate(input=f"BAC009S0150W0001.wav", hotword='')
print(time.time()-t)
print(res)
