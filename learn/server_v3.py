import websockets
import asyncio
import numpy as np
import transformers
from utils.server_utils import init_asr_model,asr,ct2_asr,init_ct2_model,init_vad_model
from utils.silero_vad.utils_vad import init_jit_model
import queue
import threading
import torch
import json

#32768

wav_queue = queue.Queue()
asr_queue = queue.Queue()

asr_res=""
asr_history_res=""

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
        asr_history_res += ct2_asr(ct2_model,ct2_processor,asr_in_data) +"|END:"+data_pack.split("|")[-1]+""+ "<br>"
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
      asr_res = ct2_asr(ct2_model,ct2_processor,asr_in_data)
      chunk_data=[]
      pass
    pass
  pass

# vad的线程
def vad_thread():
  # vad模型输入的数据
  vad_in_data=[]
  start_vad=False
  while True:
    data_pack = wav_queue.get()
    vad_in_data.append(data_pack)
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
    if isinstance(message,str):
      # 正常的数据
      print(message)
      message_obj = json.loads(message)
      if 'signal' in message_obj:
        asr_res=""
        asr_history_res=""
        vad_iterator.reset_states()
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

