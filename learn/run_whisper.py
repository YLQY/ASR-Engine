from transformers import WhisperProcessor,WhisperForConditionalGeneration
import torchaudio
import time


model_path="/home/wenet_data2/tt/asr_server/model/whisper-tiny"
#model_path="/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo"


# 加载whisper模型
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path).to('cuda')
whisper_processor = WhisperProcessor.from_pretrained(model_path)


#wav = torchaudio.load("BAC009S0150W0001.wav",normalize=False)[0][0]/32768
wav = torchaudio.load("BAC009S0150W0001.wav",normalize=False)[0][0]
print(wav)

input_features = whisper_processor(wav,return_tensors="pt").input_features.to('cuda')
print(input_features)

t = time.time()
# 输出的token
predict_ids = whisper_model.generate(input_features)
print(whisper_processor.batch_decode(predict_ids))
print(time.time()-t,"s")
