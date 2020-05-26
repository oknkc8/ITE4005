import numpy as np
from collections import defaultdict
from collections import deque

class DBSCAN:
	def __init__(self, points, eps, minpts):
		super(DBSCAN, self).__init__()
		
		self.points = points
		self.eps = eps
		self.minpts = minpts

		self.label = [0] * len(points)
		self.clusters = defaultdict(list)
		self.cnt =0

		self.adj_list = defaultdict(list)

		self.make_adj_list()

	def __len__(self):
		return len(self.clusters)

	def check_neighbor(self, i, j):
		p1 = self.points[i]
		p2 = self.points[j]

		import pdb
		#pdb.set_trace()

		if np.linalg.norm(p1 - p2) <= self.eps:
			return True
		else:
			return False

	def make_adj_list(self):
		for i in range(len(self.points)):
			for j in range(i+1, len(self.points)):
				if self.check_neighbor(i, j):
					self.adj_list[i].append(j)
					self.adj_list[j].append(i)

	def is_core(self, i):
		return (len(self.adj_list[i]) >= self.minpts)

	def bfs(self, core):
		Q = deque([core])
		self.label[core] = self.cnt
		
		while len(Q):
			now = Q.popleft()
			if self.is_core(now) == False:
				continue

			for togo in self.adj_list[now]:
				if self.label[togo] == 0:
					self.label[togo] = self.cnt
					Q.append(togo)

	def cluster(self):
		#for i in range(len(self.points)):
		for i, _ in enumerate(self.points):
			if self.label[i] != 0:
				continue
				
			if self.is_core(i):
				self.cnt += 1
				self.bfs(i)
			else:	# This point is Noise!
				self.label[i] = -1
	
		for i, L in enumerate(self.label):
			self.clusters[L].append(i)

		return self.clusters