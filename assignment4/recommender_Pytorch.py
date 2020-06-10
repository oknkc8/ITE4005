import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import argparse
from os import path
import easydict
import time
import pdb
from tqdm import tqdm

from models.MatrixFactorization_Pytorch import MatrixFactorization

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Input(train_path, test_path):
	train_data = pd.read_csv(train_path, sep='\t', header=None)
	test_data = pd.read_csv(test_path, sep='\t', header=None)

	train_data.rename(columns={0:'user_id', 1:'item_id', 2:'rating', 3:'time_stamp'}, inplace=True)
	test_data.rename(columns={0:'user_id', 1:'item_id', 2:'rating', 3:'time_stamp'}, inplace=True)

	# Remove Time Stamp Attribute
	train_data.drop('time_stamp', axis=1, inplace=True)
	test_data.drop('time_stamp', axis=1, inplace=True)

	# Make User-Item Matrix
	R = train_data.pivot_table('rating', index='user_id', columns='item_id').fillna(0)
	R = R.values

	train_data = np.array(train_data.values)
	test_data = np.array(test_data.values)

	return R, train_data, test_data

def Output(test_path, output_path, test_data, rating_result):
	file_name = path.splitext(path.basename(test_path))[0] + '.base_prediction.txt'
	file_path = path.join(output_path, file_name)

	user_id = test_data[:, 0]
	item_id = test_data[:, 1]

	rating_result[rating_result > 5] = 5.0
	rating_result[rating_result < 0] = 0
	rating_result = np.around(rating_result)
	
	with open(file_path, 'w') as file:
		for u, i, r in zip(user_id, item_id, rating_result):
			file.write(str(u) + '\t' + str(i) + '\t' + str(r) + '\n')

def train(model, R, train_data, test_data, epochs=100, lr=0.01, wd=0.0, print_log=True, log_step=10):
	user_id = torch.LongTensor(train_data[:, 0])
	item_id = torch.LongTensor(train_data[:, 1])
	rating = torch.Tensor(train_data[:, -1])

	model.train()

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

	for epoch in tqdm(range(epochs)):
		rating_predicted = model(user_id, item_id)
		
		loss = torch.sqrt(criterion(rating_predicted, rating))

		#pdb.set_trace()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if print_log and (epoch+1) % log_step == 0:
			#Validation
			with torch.no_grad():
				model.eval()
				user_id_valid = torch.LongTensor(test_data[:, 0])
				item_id_valid = torch.LongTensor(test_data[:, 1])
				rating_valid = torch.LongTensor(test_data[:, -1])

				rating_predicted = model(user_id_valid, item_id_valid)
				
				loss_valid = torch.sqrt(criterion(rating_predicted, rating_valid))
				print('Epoch %d => Loss: %.5f, Valid_Loss: %.5f' % (epoch+1, loss.item(), loss_valid.item()))

			model.train()

def test(model, test_data):
	user_id = torch.LongTensor(test_data[:, 0])
	item_id = torch.LongTensor(test_data[:, 1])
	
	model.eval()
	rating_result = model(user_id, item_id)

	return rating_result.detach().numpy()

if __name__ == "__main__":
	# argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("train_path", type=str, help="Training data file name")
	parser.add_argument("test_path", type=str, help="Test data file name")
	
	parser.add_argument("--output_path", type=str, default='./test/', help="Output file path")
	parser.add_argument("--factor", type=int, default=3, help="Latent Parameter")
	parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
	parser.add_argument("--wd", type=float, default=0.0, help="Weight Decay")
	parser.add_argument("--epochs", type=int, default=100, help="Training Epochs")
	parser.add_argument("--log_step", type=int, default=10, help="Training Epochs")
	parser.add_argument("--print_log", type=str2bool, default=False, help="Training Epochs")

	args = parser.parse_args()
	# args = easydict.EasyDict({
	# 	"train_path" : "~/Hanyang/4-1/ITE4005/assignment4/data/u1.base",
	# 	"test_path" : "~/Hanyang/4-1/ITE4005/assignment4/data/u1.test",
	# 	"output_path" : "~/Hanyang/4-1/ITE4005/assignment4/test/",
	# 	"factor" : 10,
	# 	"lr" : 0.05,
	# 	"epochs" : 50,
	# 	"log_step" : 1,
	# 	"print_log" : True
	# })

	start_time = time.time()
	
	R, train_data, test_data = Input(args.train_path, args.test_path)
	
	M = train_data[:, 0].max() + 5
	N = train_data[:, 1].max() + 5
	bias = np.mean(R[R!=0])

	model = MatrixFactorization(M, N, args.factor, bias)
	train(model,
		  R, 
		  train_data, 
		  test_data, 
		  args.epochs, 
		  args.lr, 
		  args.wd, 
		  args.print_log, 
		  args.log_step)

	rating_result = test(model, test_data)

	Output(args.test_path, args.output_path, test_data, rating_result)


	end_time = time.time()

	print('Total Running Time: %.5f' % (end_time - start_time))