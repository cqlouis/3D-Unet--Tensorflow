import argparse
import os
import tensorflow as tf
from model import Model
from configure import conf

"""This script defines hyperparameters.
"""

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='train',
						help='actions: train or predict')
	args = parser.parse_args()

	if args.option not in ['train', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train or predict")
	else:
		model = Model(conf)
		getattr(model, args.option)()

if __name__ == '__main__':
	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = '5'
	tf.logging.set_verbosity(tf.logging.INFO)   # 将 TensorFlow 日志信息输出到屏幕
	tf.app.run()  #tensorflow的程序中,在main函数下,都是使用tf.app.run()来启动，可以猜到，应该是函数入口，类似于c/c++中的main()。
