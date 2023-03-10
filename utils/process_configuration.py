import yaml
import os
from bunch import Bunch

class ConfigurationParameters:
	def __init__(self, args):
		self.args = args
		# yaml_file = self.args['config`]
		yaml_file = self.args.config

		with open(yaml_file, 'r') as config_file:
			self.config_dictionary = yaml.load(config_file, Loader=yaml.FullLoader)

		# Bunch class is subclass of dict.
		# This have update function
		self.config_namespace = Bunch(self.config_dictionary)

if __name__ == '__main__':
	ConfigurationParameters(args)