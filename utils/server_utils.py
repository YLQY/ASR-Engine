from transformers import WhisperProcessor,WhisperForConditionalGeneration
import torchaudio
import time
import transformers
import ctranslate2
from utils.silero_vad.utils_vad import init_jit_model,VADIterator
from funasr import AutoModel
from utils.tree_node import TreeNode
import numpy as np
import wave
import os
#whisper_model_path="/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo"
#whisper_model_path="/home/wenet_data2/tt/asr_server/model/whisper-tiny"

def pcm_to_wav(pcm_file, wav_file, channels=1, sample_width=2, frame_rate=16000):
  # 打开PCM文件
  with open(pcm_file, 'rb') as pcmfile:
    pcm_data = pcmfile.read()

  # 创建WAV文件
  with wave.open(wav_file, 'wb') as wavfile:
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(sample_width)
    wavfile.setframerate(frame_rate)
    wavfile.writeframes(pcm_data)

def save_to_wav(in_data,idx):
  # 将列表转换为NumPy数组，并指定数据类型为16位整数
  array = np.array([i*32768 for i in in_data], dtype=np.int16)

  # 将NumPy数组保存为二进制文件
  with open('/home/wenet_data2/tt/asr_server/uploads/wav/'+idx+".pcm", 'wb') as f:
    array.tofile(f)

  pcm_to_wav('/home/wenet_data2/tt/asr_server/uploads/wav/'+idx+".pcm", '/home/wenet_data2/tt/asr_server/uploads/wav/'+idx+".wav", channels=1, sample_width=2, frame_rate=16000)
  os.remove('/home/wenet_data2/tt/asr_server/uploads/wav/'+idx+".pcm")
  pass


# 得到热词,paraformers的
def get_hot_words():
  hot_words=""
  with open("/home/wenet_data2/tt/asr_server/uploads/hotword/hotwords.txt",'r',encoding="utf-8") as file:
    for line in file.readlines():
      line = line.strip()
      hot_words+=line+" "
      pass
    pass
  return hot_words

# 加载whisper的热词
def get_whisper_hot_words(whisper_processor):
  all_hot_words_token = []
  with open("/home/wenet_data2/tt/asr_server/uploads/hotword/hotwords.txt",'r',encoding="utf-8") as file:
    for line in file.readlines():
      line = line.strip()
      tokens = whisper_processor.tokenizer(line,skip_special_tokens=True).input_ids[2:-1]
      all_hot_words_token.append([-1]+tokens+[99999])
      pass
    pass
  root = TreeNode(-1)
  root_ori = root
  for hot_tokens in all_hot_words_token:
    root=root_ori
    # 每个热词，按照token一个一个添加
    for item in hot_tokens:
      # 根节点已经存在，不用添加
      if item == -1:
        continue
      temp = TreeNode(item,root.deep+1)
      # 添加节点
      temp = root.add_node(temp)
      root=temp
      pass
  return root_ori

# 初始化vad模型
def init_vad_model():
  vad_model = init_jit_model("/home/wenet_data2/tt/asr_server/utils/silero_vad/data/silero_vad.jit",device="cuda")
  vad_iterator = VADIterator(vad_model, sampling_rate=16000,min_silence_duration_ms=1440)
  return vad_model,vad_iterator

# 初始化阿里paraformer模型
def init_paraformer_model():
  model = AutoModel(model="/home/wenet_data2/tt/asr_server/model/paraformer-zh",device="cuda")
  return model

# 初始化阿里sensevoice模型
def init_sensevoice_model():
  model = AutoModel(model="/home/wenet_data2/tt/asr_server/model/sensevoice",device="cuda")
  return model

# 初始化asr-whisper-tiny模型
def init_asr_model():
  # 加载whisper模型
  whisper_model = WhisperForConditionalGeneration.from_pretrained("/home/wenet_data2/tt/asr_server/model/whisper-tiny").to('cuda')
  whisper_processor = WhisperProcessor.from_pretrained("/home/wenet_data2/tt/asr_server/model/whisper-tiny")
  return whisper_model,whisper_processor

# 初始化asr-whisper-large-v3-turbo模型
def init_asr_turbo_model():
  # 加载whisper模型
  whisper_model = WhisperForConditionalGeneration.from_pretrained("/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo").to('cuda')
  whisper_processor = WhisperProcessor.from_pretrained("/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo")
  return whisper_model,whisper_processor


# asr识别
def asr(whisper_model,whisper_processor,root_ori,wav):
  input_features = whisper_processor(wav,sampling_rate=16000,return_tensors="pt").input_features.to('cuda')
  predict_ids = whisper_model.generate(
    input_features=input_features,
    num_beams=5,
    max_new_tokens=255,
    hot_tree=root_ori,
    hot_words_score=1,
    length_penalty=2
  )
  res = whisper_processor.batch_decode(predict_ids,skip_special_tokens=True)[0]
  return res

def init_ct2_model():
  ct2_processor = transformers.WhisperProcessor.from_pretrained("/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo")
  ct2_model = ctranslate2.models.Whisper("/home/wenet_data2/tt/asr_server/model/whisper-large-v3-turbo-ct2",device="cuda")
  return ct2_model,ct2_processor

# ct2 asr识别
def ct2_asr(ct2_model,ct2_processor,wav):
  inputs = ct2_processor(wav, return_tensors="np", sampling_rate=16000)
  features = ctranslate2.StorageView.from_array(inputs.input_features)
  results = ct2_model.generate(features, [[50258, 50260, 50360, 50364]])
  transcription = ct2_processor.decode(results[0].sequences_ids[0])
  return transcription











