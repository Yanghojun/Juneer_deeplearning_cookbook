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
from base.data_loader_base import BaseDataLoader

class Keras_Dataloader(keras.utils.Sequence):
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

		# self.train_x = np.array([])
		# self.train_y = np.array([])

		# self.test_x = np.array([])
		# self.test_y = np.array([])

		# self.load_dataset()
		# self.preprocess_dataset()
		# self.print_dataset_details()
	

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
	# 	raise NotImplementedError()1

class LasDataLoader(DataLoader, keras.utils.Sequence):
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

	def __len__(self):
		# torch는 batchsize로 따로 조절해줄 필요 없을듯. torch Loader 클래스에 batch argument 존재
		if self.config.config_namespace.PYTORCH:
			return len(self.x_paths)

		else:
			return int(np.floor(len(self.x_paths) / self.batch_size))

	def __getitem__(self, idx):
		# batch_x_path = self.img_paths[idx * self.batch_size:(idx+1) * self.batch_size]
		# batch_x_path = self.x_paths[idx * self.batch_size:(idx+1) * self.batch_size]
		# batch_y_path = self.y_paths[idx * self.batch_size:(idx+1) * self.batch_size]

		if self.config.config_namespace.PYTORCH:
			x_path, y_path = self.x_paths[idx], self.y_paths[idx]
			img = Image.open(x_path)
			label = Image.open(y_path)
			# img, label = Image.open(x_path), Image.open(y_path)
			img, label = transforms.ToTensor(img), transforms.ToTensor(label)
			return img, label

		else:
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

class LasDataSet(Dataset):
	def __init__(self, config, train=True):
		self.config = config
		self.batch_size = self.config.config_namespace.BATCH_SIZE
		self.transform = transforms.ToTensor()

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

	def __len__(self):
		# torch는 batchsize로 따로 조절해줄 필요 없을듯. torch Loader 클래스에 batch argument 존재
		if self.config.config_namespace.PYTORCH:
			return len(self.x_paths)

		else:
			return int(np.floor(len(self.x_paths) / self.batch_size))

	def __getitem__(self, idx):
		# batch_x_path = self.img_paths[idx * self.batch_size:(idx+1) * self.batch_size]
		# batch_x_path = self.x_paths[idx * self.batch_size:(idx+1) * self.batch_size]
		# batch_y_path = self.y_paths[idx * self.batch_size:(idx+1) * self.batch_size]

		if self.config.config_namespace.PYTORCH:
			x_path, y_path = self.x_paths[idx], self.y_paths[idx]
			img = Image.open(x_path)
			label = Image.open(y_path)

			img, label = self.transform(img), self.transform(label)
			# img = transforms.ToTensor(img)
			# label = transforms.ToTensor(label)
			# img, label = transforms.ToTensor(img), transforms.ToTensor(label)
			return img, label

		else:
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

def build_las_data_loader(config, train):
	dataset = LasDataSet(config, train)
	dataloader = DataLoader(dataset, batch_size=config.config_namespace.BATCH_SIZE, shuffle=True)
	return dataloader