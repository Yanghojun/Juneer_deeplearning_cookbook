class BaseModel():
	def __init__(self, config, train_dataloader, test_dataloader):
		self.train_dataloader = train_dataloader
		self.test_dataloader = test_dataloader
		self.config = config

		self.model = self.define_model()
		
	def define_model(self):
		raise NotImplementedError

	def fit_model(self):
		raise NotImplementedError