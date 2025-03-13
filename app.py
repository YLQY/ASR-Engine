#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: zhendong.peng@mobvoi.com (Zhendong Peng)
import os
import argparse

from flask import Flask, render_template,request

parser = argparse.ArgumentParser(description='training your network')
parser.add_argument('--port', default=19999, type=int, help='port id')
args = parser.parse_args()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload_hot_words", methods=["POST"])
def upload_hot_words():
    file = request.files.get("hot_word_file")
    if file is None:# 表示没有发送文件
        return "上传失败"
    file_name = file.filename.replace(" ","")
    file.save(os.path.dirname(__file__)+'/uploads/hotword/hotwords.txt')  # 保存文件
    return "上传成功"
                        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=True)
