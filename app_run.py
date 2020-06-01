from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import tensorflow.compat.v1 as tf
from model import Config,Model
from flask import render_template
from read_utils import TextConverter, batch_generator
from flask_bootstrap import Bootstrap

# from model_att_bi import Config,Model

import os
import numpy as np
#from gevent import pywsgi
from gevent.pywsgi import WSGIServer
import logging
tf.reset_default_graph()
app = Flask(__name__)
# CORS(app)
Bootstrap(app)
vocab_file = '/data/vocabs'
model_dir = '/models/lstm_c'

#m = Model(Config)

@app.route('/')
def hello():
  return render_template('test.html')

def isAllChinese(s):
 for c in s:
  if not('\u4e00' <= c <= '\u9fa5'):
   return False
 return True

@app.route('/cp', methods=['POST'])
def cp():
    in_str=request.form.get('in_str');
    str_in=in_str
    print("input_str="+in_str)
    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
   # 加载上一次保存的模型
    m = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        m.load(checkpoint_path)
    in_str = ' '.join(in_str)
    in_str = in_str.split()
    converter = TextConverter(vocab_dir='data/vocabs', max_vocab=Config.vocab_size, seq_length=Config.seq_length)
    en_arr, arr_len = converter.text_en_to_arr(in_str)
    test_g = [np.array([en_arr, ]), np.array([arr_len, ])]
  #  output_ids = m.test(test_g, model_path, converter)
    if len(in_str) == 0 or len(in_str)> 50  or isAllChinese(str_in)==False:
        output = u'您的输入有误'
        print(len(in_str))
    else:
        output_ids =m.test(test_g, model_path, converter)
        output = converter.arr_to_text(output_ids)
      #  output = ''.join(output.split(' '))

       # in_str = converter.arr_to_text(in_str)
    print('上联：%s；下联：%s' % (in_str, output))
   #return jsonify({'output': output,'in_str': in_str})
    return jsonify({"resultCode": 200, "error": "", "in_str" : str_in,"output" : output})
   # return render_template('test.html', in_str=in_str,output=output)

if __name__ == '__main__':
    app.run()
