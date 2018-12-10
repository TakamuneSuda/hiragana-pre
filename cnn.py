import tensorflow as tf
from PIL import Image
from model import cnn_model_fn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mode',
                    help='tarin or predict',
                    metavar='',
                    dest='mode',
                    choices=['train', 'predict'],
                    required=True)

parser.add_argument('--image_file',
                    help='predict image file',
                    metavar='',
                    dest='image_file',
                    required=False)

model_dir = './model/etl8g_convnet_model'

# create Estimator object
etl8g_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=model_dir)

args = parser.parse_args()
if args.mode == 'predict':
    if args.image_file is None:
        raise ValueError('Please set predict image file path '
                         'using --image_file.')

    image = Image.open(args.image_file)
    predict_data = np.asarray(image).reshape(1, 1024).astype(np.float32)/255

    # set input data
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        shuffle=False)

    predict_results = list(etl8g_classifier.predict(predict_input_fn))
    result = predict_results[0]['classes']

    classmapping = pd.read_csv('./classmapping.csv', usecols=['ひらがな'], encoding='cp932')
    print('画像のひらがなは「{}」です'.format(classmapping.iloc[result].ひらがな))


elif args.mode == 'train':
    # load_data
    hira = np.load('./hira.npz')
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        hira['data'], hira['labels'], stratify=hira['labels'], random_state=0)

    # hooks
    tf.logging.set_verbosity(tf.logging.INFO)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=500,
        output_dir=model_dir,
        scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

    # set input data
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    # train and evaluate
    for i in range(60):
        etf8g_classifier.train(
            input_fn=train_input_fn,
            steps=500,
            hooks=[logging_hook, summary_hook])

        eval_results = etf8g_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
