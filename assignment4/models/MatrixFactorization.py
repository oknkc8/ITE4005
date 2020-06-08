import numpy as np
import time
import pdb

class MatrixFactorization:
	def __init__(self, R, f, lr, reg, epochs, log_step, print_log, test_data):
		super(MatrixFactorization, self).__init__()

		"""
		R : Rating Matrix
		f : Latent Parameter, R[M x N] = U[M x f] X (V[f x N]^T)
		lr : Learning Rate
		reg : Regularization Strength
		epochs : Training Epochs
		log_step : Period of Printing Log
		print_log : Print Training loss
		"""
		self.R = R.values
		self.M, self.N = R.shape # M : # of users
								 # N : # of items
		self.f = f
		self.lr = lr
		self.reg = reg
		self.epochs = epochs
		self.log_step = log_step
		self.print_log = print_log
		self.test_data = test_data

		self.user_id_idx = {}
		self.item_id_idx = {}

		for idx, user_id in enumerate(R.index):
			self.user_id_idx[user_id] = idx
		for idx, item_id in enumerate(R.columns):
			self.item_id_idx[item_id] = idx

	def train(self):
		"""
		R = U X V^T
		"""
		self.U = np.random.normal(scale=0.1,size=(self.M, self.f))
		self.V = np.random.normal(scale=0.1,size=(self.N, self.f))
		# self.U = np.zeros((self.M, self.f))
		# self.V = np.zeros((self.N, self.f))

		self.bias_U = np.zeros(self.M)
		self.bias_V = np.zeros(self.N)
		self.bias = np.mean(self.R[self.R != 0])

		ex_cost_valid = 10

		idx_i, idx_j = self.R.nonzero()
		idxs = np.array([idx_i, idx_j]).T

		for epoch in range(self.epochs):
			start_time = time.time()

			np.random.shuffle(idxs)
			for idx in idxs:
				i, j = idx
				self.update_value(i, j)


			# for i in range(self.M):
			# 	#pdb.set_trace()
			# 	for j in range(self.N):
			# 		if self.R[i][j] > 0:
			# 			self.update_value(i, j)
			
			cost = self.get_loss()
			cost_valid = self.get_valid_loss()
			end_time = time.time()

			if ex_cost_valid < cost_valid:
				print('Warning! => Overfitting!!')

			ex_cost_valid = cost_valid

			if self.print_log and (epoch+1) % self.log_step == 0:
				print('Epoch: %d => cost: %.5f, valid: %.5f, time: %.5f' % (epoch+1, cost, cost_valid, end_time - start_time))

	
	def update_value(self, i, j):
		error = self.R[i][j] - self.predict_value(i, j)

		self.bias_U[i] += self.lr * (error - self.reg * self.bias_U[i])
		self.bias_V[j] += self.lr * (error - self.reg * self.bias_V[j])
		# self.bias_U[i] += self.lr * (error)
		# self.bias_V[j] += self.lr * (error)
		#pdb.set_trace()

		dU = (error * self.V[j, :]) - (self.reg * self.U[i, :])
		dV = (error * self.U[i, :]) - (self.reg * self.V[j, :])

		self.U[i, :] += self.lr * dU
		self.V[j, :] += self.lr * dV

	def predict_value(self, i, j):
		ret = self.bias + self.bias_U[i] + self.bias_V[j] + np.dot(self.U[i, :], self.V[j, :].T)
		# ret = min(ret, 5)
		# ret = max(ret, 0)
		return ret

	def get_loss(self):
		R_predicted = self.get_predicted_R()
		idx_i, idx_j = self.R.nonzero()
		n = idx_i.size

		cost = 0
		for i, j in zip(idx_i, idx_j):
			cost += (self.R[i][j] - R_predicted[i][j]) ** 2
		
		return np.sqrt(cost / n)
	
	def get_predicted_R(self):
		R_predicted = self.bias + np.expand_dims(self.bias_U, -1) + np.expand_dims(self.bias_V, 0) + np.dot(self.U, self.V.T)
		return R_predicted

	def test(self):
		test_user_id = self.test_data[:, 0]
		test_item_id = self.test_data[:, 1]
		R = self.get_predicted_R()

		ret = []
		#pdb.set_trace()
		for user_id, item_id in zip(test_user_id, test_item_id):
			if item_id in self.item_id_idx:
				user_idx = self.user_id_idx[user_id]
				item_idx = self.item_id_idx[item_id]

				r = max(0, R[user_idx][item_idx])
				r = min(5, r)
				ret.append(r)
			else:
				ret.append(self.bias)
		
		return np.array(ret)

	def get_valid_loss(self):
		test_rating = self.test_data[:, -1]
		test_result = self.test()
		
		rmse = 0
		for i, _ in enumerate(test_rating):
			rmse += (test_rating[i] - test_result[i]) ** 2
		rmse = np.sqrt(rmse / len(test_rating))
		
		return rmse
