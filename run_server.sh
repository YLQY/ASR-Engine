ps -ef | grep server.py | grep -v grep | cut -c 9-15 | xargs kill -s 9
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/asr_server/lib/python3.10/site-packages/nvidia/cudnn/lib
python3 server.py
