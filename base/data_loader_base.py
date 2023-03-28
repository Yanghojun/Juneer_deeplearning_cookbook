# pytorch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# keras
from keras.utils import load_img
import keras.utils

# etc..
from sklearn.model_selection import train_test_split
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

class BaseDataLoader():
	def __init__(self, config, train=True):
		self.config = config
		self.batch_size = self.config.config_namespace.BATCH_SIZE
		self.train = train
		# Set data directory pyat
		if self.train:
			path = self.config.config_namespace.TRAIN_DATA_PATH
		else:
			path = self.config.config_namespace.TEST_DATA_PATH
		self.dir_path = path

		# 어떤 객체를 생성하면 ~~ 이 정보를 다 줘야할 거 같은데..?



		# self.data_loader = self.build_data_loader(self.config.config_namespace.PYTORCH)

	def build_data_loader(self, pytorch=True):
		""" This function works to build selected framework(pytorch and keras) from cfg file.

		Args:
			pytorch (bool, optional): Defaults to True.

		Returns:
			torch.utils.data.DataLoader | keras.utils.Sequence.: data_loader
		"""

		if pytorch:
			data_loader = Pytorch_DataLoader(self.config, self.train)

		else:
			data_loader = Keras_DataLoader(self.config, self.train)

		return data_loader

	def __len__(self):
		raise NotImplementedError
	
	def __getitem__(self):
		raise NotImplementedError


		
class Keras_DataLoader(keras.utils.Sequence):
	def __init__(self, config, train=True):
		self.config = config
		self.batch_size = self.config.config_namespace.BATCH_SIZE

		if train:
			dir_path = self.config.config_namespace.TRAIN_DATA_PATH
			# self.img_paths = glob(os.path.join(self.config.config_namespace.TRAIN_DATA_PATH, '*.png'))
		else:
			dir_path = self.config.config_namespace.TEST_DATA_PATH
			# self.img_paths = glob(os.path.join(self.config.config_namespace.TEST_DATA_PATH, '*.png'))
		file_list = os.listdir(dir_path)
		self.x_paths = [os.path.join(dir_path, file) for file in file_list if '_denoised.png' not in file]
		self.y_paths = [os.path.join(dir_path, file) for file in file_list if '_denoised.png' in file]
		self.x_paths.sort(), self.y_paths.sort()

	def __getitem__(self, idx):
		# batch_x_path = self.img_paths[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_x_path = self.x_paths[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_y_path = self.y_paths[idx * self.batch_size:(idx+1) * self.batch_size]

		x_li = []
		y_li = []
		for img_path in batch_x_path:
			img = load_img(img_path)
			img = np.array(img)
			img = img[:, :, :self.config.config_namespace.CHANNELS]
			img = img / 255.0
			x_li.append(img)
		
		for img_path in batch_y_path:
			img = load_img(img_path)
			img = np.array(img)
			img = img[:, :, :self.config.config_namespace.CHANNELS]
			img = img / 255.0
			y_li.append(img)

			# cv_img = np.where(img < self.config.config_namespace.THRESHOLD, 0, img)
			# cv_img = cv_img / 255.0
			# y_li.append(cv_img)

		x = np.array(x_li)
		y = np.array(y_li)

		return x, y

	def __len__(self):
		return int(np.floor(len(self.x_paths) / self.batch_size))

	# shuffle 기능은 나중에..?
	def on_epoch_end(self):
		pass

	# def __iter__(self):
	# 	pass

	# def load_dataset(self):
	# 	raise NotImplementedError()

	# def print_dataset_details(self):
	# 	raise NotImplementedError()

	# def preprocess_dataset(self):
	# 	raise NotImplementedError()