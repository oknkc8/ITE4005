import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import pdb

class MatrixFactorization(nn.Module):
	def __init__(self, M, N, f, bias):
		super(MatrixFactorization, self).__init__()

		self.user_emb = nn.Embedding(M, f)
		self.item_emb = nn.Embedding(N, f)
		self.user_bias_emb = nn.Embedding(M, 1)
		self.item_bias_emb = nn.Embedding(N, 1)
		self.bias = bias
		
		self.user_emb.weight.data.normal_(0, 0.5)
		self.item_emb.weight.data.normal_(0, 0.5)
		self.user_bias_emb.weight.data.fill_(0.0)
		self.item_bias_emb.weight.data.fill_(0.0)

	def forward(self, users, items):
		#pdb.set_trace()
		U = self.user_emb(users)
		V = self.item_emb(items)
		bias_U = self.user_bias_emb(users).squeeze()
		bias_V = self.item_bias_emb(items).squeeze()

		rating = (U * V).sum(1) + bias_U + bias_V + self.bias

		return rating