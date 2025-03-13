import websockets
import asyncio
import numpy as np
import transformers
from utils.server_utils import init_asr_model,asr,ct2_asr,init_ct2_model,init_vad_model,init_asr_turbo_model,init_paraformer_model,init_sensevoice_model,get_hot_words,get_whisper_hot_words,save_to_wav
import uuid
from utils.silero_vad.utils_vad import init_jit_model
import queue
import threading
import torch
import json

#32768

wav_queue = queue.Queue()
asr_queue = queue.Queue()

# 中间上屏的结果
asr_res=""
# 最终上屏的结果
asr_history_res=""

# 选择asr引擎
asr_engine=""

# 热词
hot_words=""

# whisper的热词
whisper_tree_node=None

# sessionid
sid=""

# asr的选择
def lainspeech_asr(asr_in_data):
  speech_res=""
  if asr_engine=="whisper-tiny":
    speech_res = asr(whisper_model,whisper_processor,whisper_tree_node,asr_in_data)
    pass
  elif asr_engine=="whisper-large-v3-turbo":
    speech_res = asr(whisper_model_turbo,whisper_processor_turbo,whisper_tree_node,asr_in_data)
    pass
  elif asr_engine=="whisper-large-v3-turbo-ct2":
    speech_res = ct2_asr(ct2_model,ct2_processor,asr_in_data)
    pass
  elif asr_engine=="paraformer":
    speech_res = paraformer_model.generate(input=np.array(asr_in_data), hotword=hot_words)[0]['text']
    pass
  elif asr_engine=="sensevoice":
    speech_res = sensevoice_model.generate(input=np.array(asr_in_data))[0]['text'].replace("<|Speech|><|woitn|>","").replace("<|zh|>")
    pass
  else:
    speech_res="没有指定的asr引擎"    

  return speech_res


# asr的线程
def asr_thread():
  global asr_res
  global asr_history_res
  asr_in_data=[]
  chunk_data=[]
  while True:
    data_pack = asr_queue.get()
    # 如果给的是字符串数据
    if isinstance(data_pack,str):
      # 开始
      if 'vad_start' in data_pack:
        asr_history_res += "START:"+data_pack.split("|")[-1]+"|"
        pass
      # vad结束
      if 'vad_end' in data_pack:
        asr_res = ""
        asr_history_res += lainspeech_asr(asr_in_data) +"|END:"+data_pack.split("|")[-1]+""+ "<br>"
        save_to_wav(asr_in_data,sid+"_END_"+data_pack.split("|")[-1])
        chunk_data=[]
        asr_in_data=[]
        pass
      continue
    # asr数据 
    asr_in_data.append(data_pack)
    chunk_data.append(data_pack)
    # 32ms * 20
    if len(chunk_data) == 512*20:
      # 中间上屏
      asr_res = lainspeech_asr(asr_in_data)
      chunk_data=[]
      pass
    pass
  pass

# vad的线程
def vad_thread():
  # vad模型输入的数据
  vad_in_data=[]
  start_vad=False
  first_chunk_vad=[]
  while True:
    data_pack = wav_queue.get()
    vad_in_data.append(data_pack)
    first_chunk_vad.append(data_pack)
    # 防止vad切割到开头的语音，加上的vad前面语音的缓存
    if len(first_chunk_vad) == 512*5:
      first_chunk_vad = first_chunk_vad[1:]
      pass
    # 现在有512个采样点，执行vad的模型
    if len(vad_in_data) == 512:
      # vad模型的推理
      speech_dict = vad_iterator(torch.tensor(vad_in_data,dtype=torch.float).to('cuda'), return_seconds=True)
      if speech_dict:
        print(speech_dict)
        # 音频的开始
        if 'start' in speech_dict:
          start_vad=True
          asr_queue.put("vad_start|"+str(speech_dict['start']))
          [asr_queue.put(i) for i in first_chunk_vad[:-1]]
          pass
        # 音频的结束
        if 'end' in speech_dict:
          start_vad=False
          [asr_queue.put(i) for i in vad_in_data]
          asr_queue.put("vad_end|"+str(speech_dict['end']))
          pass
      # vad是否开始了
      if start_vad:
        [asr_queue.put(i) for i in vad_in_data]
      vad_in_data=[]
      pass
    pass
  pass

async def echo(websocket):
  async for message in websocket:
    # 语音识别结果
    global asr_res
    global asr_history_res
    global asr_engine
    global hot_words
    global whisper_tree_node
    global sid
    if isinstance(message,str):
      # 正常的数据
      print(message)
      message_obj = json.loads(message)
      if 'signal' in message_obj:
        if message_obj['signal'] == 'start':
          asr_res=""
          asr_history_res=""
          vad_iterator.reset_states()
          hot_words=get_hot_words()
        if "asr_engine" in message_obj:
          # 这个session的uuid
          sid = str(uuid.uuid4())
          asr_engine=message_obj["asr_engine"]
          if asr_engine=="whisper-large-v3-turbo":
            whisper_tree_node = get_whisper_hot_words(whisper_processor_turbo)
          if asr_engine=="whisper-tiny":
            whisper_tree_node = get_whisper_hot_words(whisper_processor)
          pass
        pass 
      pass
    # 可能是音频数据
    else:
      audio_data = np.frombuffer(message,dtype=np.int16)/32768
      [wav_queue.put(i) for i in audio_data]
      #asr_res = asr(whisper_model,whisper_processor,audio_data)
      #asr_res = ct2_asr(ct2_model,ct2_processor,audio_data)
      pass

    await websocket.send('{"type":"final_result","res":"'+asr_history_res+'<br>'+asr_res+'"}')
  pass

# 初始化模型数据
vad_model,vad_iterator = init_vad_model()
vad_iterator.reset_states()
whisper_model,whisper_processor=init_asr_model()
whisper_model_turbo,whisper_processor_turbo=init_asr_turbo_model()
paraformer_model = init_paraformer_model()
sensevoice_model = init_sensevoice_model()
ct2_model,ct2_processor=init_ct2_model()

# vad线程
vad_thread = threading.Thread(target=vad_thread,args=())
# asr线程
asr_thread = threading.Thread(target=asr_thread,args=())
# 开启vad线程
vad_thread.start()
# 开启asr线程
asr_thread.start()
print("server_start")
start_server = websockets.serve(echo,"0.0.0.0",10086)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

