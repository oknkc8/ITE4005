import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import argparse
from os import path
import easydict
import time

from models.MatrixFactorization import MatrixFactorization

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

	# Make Pivot Table
	rating = train_data.pivot_table('rating', index='user_id', columns='item_id').fillna(0)

	test_data = np.array(test_data.values)
	train_data = np.array(train_data)

	return rating, train_data, test_data

def Output(test_path, output_path, test_data, rating_result):
	file_name = path.splitext(path.basename(test_path))[0] + '.base_prediction.txt'
	file_path = path.join(output_path, file_name)

	user_id = test_data[:, 0]
	item_id = test_data[:, 1]
	
	with open(file_path, 'w') as file:
		for u, i, r in zip(user_id, item_id, rating_result):
			file.write(str(u) + '\t' + str(i) + '\t' + str(r) + '\n')

def get_RMSE(train_rating, train_result, test_rating, test_result):
	rmse = 0
	for i, _ in enumerate(train_rating):
		rmse += (train_rating[i] - train_result[i]) ** 2
	rmse = np.sqrt(rmse / len(train_rating))
	print('Training RMSE: %.6f' % (rmse))

	rmse = 0
	for i, _ in enumerate(test_rating):
		rmse += (test_rating[i] - test_result[i]) ** 2
	rmse = np.sqrt(rmse / len(test_rating))
	print('Test RMSE: %.6f' % (rmse))

if __name__ == "__main__":
	# argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("train_path", type=str, help="Training data file name")
	parser.add_argument("test_path", type=str, help="Test data file name")
	
	parser.add_argument("--output_path", type=str, default='./test/', help="Output file path")
	parser.add_argument("--factor", type=int, default=3, help="Latent Parameter")
	parser.add_argument("--lr", type=float, default=0.01, help="SGD Learning Rate")
	parser.add_argument("--reg", type=float, default=0.02, help="Regularization Strength")
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
	# 	"reg" : 0.001,
	# 	"epochs" : 50,
	# 	"log_step" : 1,
	# 	"print_log" : True
	# })

	start_time = time.time()
	
	rating, train_data, test_data = Input(args.train_path, args.test_path)

	model = MatrixFactorization(rating, args.factor, args.lr, args.reg, args.epochs, args.log_step, args.print_log, test_data)
	model.train()
	rating_result = model.test()
	#rating_result = model.test(test_data[:, :-1])
	#train_rating_result = model.test(train_data[:, :-1])

	Output(args.test_path, args.output_path, test_data, rating_result)

	#get_RMSE(train_data[:, -1], train_rating_result, test_data[:, -1], rating_result)

	end_time = time.time()

	print('Total Running Time: %.5f' % (end_time - start_time))