mport collections
import json
import math
import os
import random

import numpy as np
import tensorflow as tf

import modeling
import optimization
import tokenization
import modeling
import optimization
import tokenization
import six 
import run_classifier_online

import sys 

ROOT = os.getcwd()
BERT_BASE_DIR = os.path.join(ROOT, 'uncased_L-12_H-768_A-12')
SQUAD_DIR = os.path.join(ROOT, 'squad-1.1')
OUT = os.path.join(ROOT, 'out')

sys.path.append(os.path.join(ROOT, "bert"))
    
os.environ['PYTHONPATH'] = os.path.join(ROOT, "bert")
os.environ['BERT_BASE_DIR'] = BERT_BASE_DIR
os.environ['SQUAD_DIR'] = SQUAD_DIR
os.environ['OUT'] = OUT 
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
#
# define some constants used by the model
#
MAX_SEQ_LENGTH = 128 
EVAL_BATCH_SIZE = 8 
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30
MAX_QUERY_LENGTH = 64
DOC_STRIDE = 128 

PREDICT_BATCH_SIZE = 4 

VOCAB_FILE = os.path.join(BERT_BASE_DIR, 'vocab_online.txt')
CONFIG_FILE = os.path.join(BERT_BASE_DIR, 'bert_config_online.json')
CHECKPOINT = os.path.join(OUT, 'model.ckpt-7000')

# checkpoint file path:
# model_checkpoint_path: "model.ckpt-7000"
# all_model_checkpoint_paths: "model.ckpt-7000"

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)

tf.logging.set_verbosity("WARN")

# touch flags
FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')
run_config = tf.contrib.tpu.RunConfig(model_dir=OUT, tpu_config=None)

print(CHECKPOINT)
model_fn = run_classifier_online.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=2,
    init_checkpoint=CHECKPOINT,
    learning_rate=0,
    num_train_steps=0,
    num_warmup_steps=0,
    use_tpu=False,
    use_one_hot_embeddings=False)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    predict_batch_size=EVAL_BATCH_SIZE,
    export_to_tpu=False)

#estimator = tf.contrib.tpu.TPUEstimator(
#    use_tpu=False,
#    model_fn=model_fn,
#    config=run_config,
#    train_batch_size=32,
#    eval_batch_size=8,
#    predict_batch_size=4,
#    export_to_tpu=False)

# Export the model
def serving_input_fn():
    receiver_tensors = {
        'input_ids': tf.placeholder(dtype=tf.int64, shape=[64, MAX_SEQ_LENGTH], name='input_ids'),
        'input_mask': tf.placeholder(dtype=tf.int64, shape=[64, MAX_SEQ_LENGTH], name='input_mask'),
        'segment_ids': tf.placeholder(dtype=tf.int64, shape=[64, MAX_SEQ_LENGTH], name='segment_ids')
        #'label_ids': tf.placeholder(dtype=tf.int64, shape=[64, 1], name='label_ids')
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

path = estimator.export_savedmodel(os.path.join(OUT, "export"), serving_input_fn)
os.environ['LAST_SAVED_MODEL'] = path.decode('utf-8')
