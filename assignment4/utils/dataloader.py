import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

class Recommender_Dataset(Dataset):
	def __init__(self, input_path):
		data = pd.read_csv(input_path, sep='\t', header=None)
		data.rename(columns={0:'user_id', 1:'item_id', 2:'rating', 3:'time_stamp'}, inplace=True)
		data.drop('time_stamp', axis=1, inplace=True)
		data = np.array(data.values)

		self.user_id = torch.LongTensor(data[:, 0])
		self.item_id = torch.LongTensor(data[:, 1])
		self.rating = torch.Tensor(data[:, -1])

	def __len__(self):
		return len(self.user_id)
	
	def __getitem__(self, index):
		return self.user_id[index], self.item_id[index], self.rating[index]