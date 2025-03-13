import websockets
import asyncio
import numpy as np
from utils.server_utils import init_asr_model,asr
#32768
data_all=b''

i = 16000/3
l = 1
async def echo(websocket):
  global data_all
  global l
  async for message in websocket:
    # 语音识别结果
    asr_res=""
    if isinstance(message,str):
      # 正常的数据
      print(message)
      data_all=b''
      l = 1
      if message=='{"signal":"end"}':
        data_all=b''
        pass
      pass
    # 可能是音频数据
    else:
      data_all += message
      audio_data = np.frombuffer(data_all,dtype=np.int16)/32768
      if len(audio_data) > i*l:
        asr_res = asr(whisper_model,whisper_processor,audio_data)
        l += 1

      pass

    await websocket.send('{"type":"final_result","res":"'+asr_res+'"}')
  pass

whisper_model,whisper_processor=init_asr_model() 

print("server_start")
start_server = websockets.serve(echo,"0.0.0.0",10086)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

