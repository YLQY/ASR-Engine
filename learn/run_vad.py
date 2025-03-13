from silero_vad_512.src.silero_vad.utils_vad import get_speech_timestamps,save_audio,  read_audio,  VADIterator,  collect_chunks,init_jit_model
from pprint import pprint

vad_model="/home/wenet_data2/tt/asr_server/learn/silero_vad_512/src/silero_vad/data/silero_vad.jit"

model = init_jit_model(vad_model)
print(model)


SAMPLING_RATE=16000
wav = read_audio('BAC009S0150W0001.wav', sampling_rate=SAMPLING_RATE)
print(wav)
#-----------------------------------------------------------------
# get speech timestamps from full audio file
#speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
#pprint(speech_timestamps)
#-----------------------------------------------------------------
vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)

# 32ms
window_size_samples = 512
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    print(len(chunk))
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    print(speech_dict,len(chunk))
    #if speech_dict:
    #    print(speech_dict, end='\n')
vad_iterator.reset_states()

