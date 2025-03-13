import websockets
import asyncio
import numpy as np
from utils.server_utils import init_asr_model,asr

data_all=b''

async def echo(websocket):
  global data_all
  async for message in websocket:
    # 语音识别结果
    asr_res=""
    if isinstance(message,str):
      # 正常的数据
      print(message)
      if message=='{"signal":"end"}':
        audio_data = np.frombuffer(data_all,dtype=np.int16)/32768
        asr_res = asr(whisper_model,whisper_processor,audio_data)
        data_all=b''
        print(asr_res)
        pass
      pass
    # 可能是音频数据
    else:
      data_all += message
      audio_data = np.frombuffer(message,dtype=np.int16)
      pass

    await websocket.send('{"type":"final_result","res":"'+asr_res+'"}')
  pass

whisper_model,whisper_processor=init_asr_model() 

print("server_start")
start_server = websockets.serve(echo,"0.0.0.0",10086)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

