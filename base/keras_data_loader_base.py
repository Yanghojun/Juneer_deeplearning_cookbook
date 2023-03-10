import numpy as np

class DataLoader(object):
	def __init__(self, config):
		self.config = config

		self.train_x = np.array([])
		self.test_y = np.array([])

		self.load_dataset()
		self.preprocess_dataset()
		self.print_dataset_details()
	
	def load_dataset(self):
		raise NotImplementedError()

	def print_dataset_details(self):
		raise NotImplementedError()

	def preprocess_dataset(self):
		raise NotImplementedError()