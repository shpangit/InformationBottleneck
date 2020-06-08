import numpy as np
import torch
import sklearn.preprocessing as pproc

class IB(object):
	"""
	One dimensional information bottleneck.
	For continuous data, discretization is processed.
	"""
	def __init__(self,dim_hidden = 2,data_type='discrete',beta = 0.1, tol = 10**-15,max_iter = 500, seed=None,check_result=True):
		self.dim_hidden = dim_hidden
		self.dtype = data_type
		self.beta = beta
		self.max_iter = max_iter
		self.tol = tol
		self.hist_converge = []
		self.check_res = check_result

		np.random.seed(seed)

	def discretize_data(self,X,strategy='qunatile',n_bins=5):
		X_disc = pproc.KBinsDiscretizer(n_bins=n_bins,strategy=strategy,encode='ordinal').fit_transform(X)
		return X_disc

	def data_to_distribution(self,X,return_details = False):
		x_supp = np.unique(X)
		nx = len(x_supp)
		px = np.zeros(nx)
		N = X.shape[0]

		for i,x in enumerate(x_supp):
			prob = np.where(X==x,1,0).sum()/N
			px[i] = prob

		if return_details:
			return px,nx,x_supp
		else:
			return px

	def calculate_pxy(self,X,Y):
		"""
		calculate distribution p(x) & p(x,y) & p(y|x)
		Input : numpy array, X Y unidimensional data
		Returns : numpy array, distribution list
		"""

		N = X.shape[0] # sample size. It must be same for both X and Y data

		px,nx,x_supp = self.data_to_distribution(X,return_details=True)
		self.px = px
		self.dim_x = nx
		# p_xy distribution
		y_supp = np.unique(Y)
		ny = len(y_supp)
		pxy = np.zeros((nx,ny))

		for i,x in enumerate(x_supp):
			for j,y in enumerate(y_supp):
				prob = np.where(np.logical_and(X==x,Y==y),1,0).sum()/N
				pxy[i,j] = prob

		self.pxy = pxy
		self.dim_y = ny

		px_reshape = np.tile(px.reshape((-1,1)),(1,ny)) # [p(x1),...,p(xn)] ny times to get matrix
		py_x = pxy/px_reshape
		py_x = py_x.T

		self.py_x = py_x

	def calculate_KL(self,P,Q,smoothing = 0.00001):
		if P.shape[0]!=Q.shape[0]:
			raise ValueError ('Distribution data must be same length')

		res = 0
		if P.ndim==1 and Q.ndim==1:
			res = (P*(np.log(P)-np.log(Q))).sum()

		elif P.ndim==2 and Q.ndim==2:
			py_x = P[:,:,np.newaxis]
			py_x = np.tile(py_x,(1,1,Q.shape[1]))

			py_z = Q[:,np.newaxis,:]
			py_z = np.tile(py_z,(1,P.shape[1],1))

			log_py_xz = np.log((py_x+smoothing)) - np.log((py_z+smoothing)) # for non 0 division.

			res = np.sum(py_x * log_py_xz,axis=0) # KL-div matrix with Mij = Dkl[p(y|xi)||p(y|zj)]

		return res

	def initialize_z(self):
		"""
		"""
		pz_x = np.random.uniform(low = 0.35,high = 0.75, size= (self.dim_hidden,self.dim_x)) # Z as row and X as column
		norm_pz_x = pz_x.sum(axis=1,keepdims=True)
		pz_x /= norm_pz_x

		pz = (self.px * pz_x).sum(axis = 1)
		
		py_z = pz_x.dot(self.pxy)
		pz_reshape = np.tile(pz.reshape(-1,1),(1,self.dim_y))
		py_z = py_z/pz_reshape
		py_z = py_z.T

		self.pz_x = pz_x
		self.pz = pz
		self.py_z = py_z

	def update(self,pz_x,pz,py_z,smoothing=0.00001):

		# update p(z|x)
		kl_div = self.calculate_KL(self.py_x,py_z)
		pz_reshape = np.tile(pz.reshape((-1,1)),(1,self.dim_x))
		new_pz_x = pz_reshape * np.exp(-self.beta * kl_div.T)
		z_norm = np.sum(new_pz_x,axis = 0,keepdims=True)
		new_pz_x =np.where(z_norm!=0,new_pz_x/z_norm,0)

		# THIS UPDATE RULE CAN BE BORKEN WHEN BETA IS TOO LARGE.
		# WE OBTAIN REALLY SMALL VALUES FOR UNNORM-PROBABILITES.


		# update p(z)
		new_pz = np.dot(new_pz_x,self.px)

		# update p(y|z)
		new_y_z = new_pz_x.dot(self.pxy)
		new_pz_reshape = np.tile(new_pz.reshape(-1,1),(1,self.dim_y))
		new_py_z = new_y_z/(new_pz_reshape)
		new_py_z = new_py_z.T

		# Save the old distribution
		self.pz_x_old = pz_x

		return new_pz_x,new_pz,new_py_z

	def calculate_JS(self,P,Q,pi1=0.5,pi2=0.5):
		# converged if the JS divergence is small enought
		if P.shape != Q.shape :
			raise ValueError('for JS-div, input data must be same shape')
		if pi1+pi2 !=1:
			raise ValueError('Sum of pis must be equal to 1')

		p_tilda = pi1*P + pi2*Q

		JS = pi1 * self.calculate_KL(P,p_tilda) + pi2 * self.calculate_KL(Q,p_tilda)

		return np.diagonal(JS)

	def convergence(self,pi1=0.5,pi2=0.5):
		P = self.pz_x_old
		Q = self.pz_x
		JS = self.calculate_JS(P,Q,pi1,pi2)
		max_js = np.max(JS)
		self.hist_converge.append(max_js)
		# print(max_js)
		return (max_js < self.tol)

	def fit(self,X,Y):

		# Calculate data statistics
		self.calculate_pxy(X,Y)

		# Initialize latent Variable Z
		self.initialize_z()

		converge = False

		if self.check_res: # initializing other variables if we want to check the results.
			self.initialize_check(Y)

		for iter in range(self.max_iter):

			self.pz_x,self.pz,self.py_z = self.update(self.pz_x,self.pz,self.py_z)
			if self.check_res:
				self.tracking_results()

			converge = self.convergence()
			if converge :
				break

		self.clustering()

	# MI computation for check the results

	def MI(self,P,Q,PQ,smoothing=0.00001):
		"""
		P,Q,two distribution (1d array)
		PQ joint distribution (2d array)
		"""
		PQ_prod = P.reshape((-1,1)).dot(Q.reshape((1,-1))) #distribution (P x Q)
		
		MI_mat = PQ * (np.log(PQ+smoothing) - np.log(PQ_prod+smoothing))

		return MI_mat.sum()


	def initialize_check(self,Y):
		self.py = self.data_to_distribution(Y)
		self.I_xy = self.MI(self.px,self.py,self.pxy)
		self.track = {'I_yz':[],
						'I_xz':[],
						}
		self.L = [] # Objective function


	def tracking_results(self,type='MI'):
		"""
		Checking results.
		For now we see the evolution of MI. We could add more checking.
		"""

		# CASE1 : tracking I(Z,Y).
		# If beta is large enough, I(Z,Y) = I(X,Y)
		pz = self.pz
		py = self.py
		pyz = self.py_z * self.pz
		I_yz = self.MI(pz,py,pyz.T)
		
		self.track['I_yz'].append(I_yz)

		# CASE2 : tracking I(X,Z).
		# If beta is null, I(X,Z) = I(Z,Y) = 0

		px = self.px
		pzx = self.pz_x * px

		I_xz = self.MI(pz,px,pzx)

		self.track['I_xz'].append(I_xz)

		self.L.append(I_xz - self.beta * I_yz)


	# TO DO :
	# Clustering results + sIB algorithm.

	def clustering(self):

		#compute P(X|Z)
		# Linking X-Z
		pz_x = self.pz_x
		px = self.pxy
		pz = self.pz
		px = self.px

		nz,nx = pz_x.shape

		if nz!=len(pz) and nx!=len(px):
			print("pz_x has different dimension than pz and px")

		px = np.tile(px.reshape((-1,1)),(1,nz))
		# matrix format of px [nx,nz]
		pz = np.tile(pz.reshape((1,-1)),(nx,1)) 

		px_z = (pz_x.T) * (px/pz)

		self.px_z = px_z

	def Error_checking():
		"""
		TODO
		Check if there is errory like, sum of distribution != 1
		"""

		np.all(self.pz_x.sum(0)) == 1
		np.all(self.pz.sum(0)) == 1
		np.all(self.py_x.sum(0)) == 1


	def contributions(self):
		"""
		See the equation 4.1 from The Information Bottleneck: Theory and Applications, Slonim
		Mesauring the contribution of the latent variables.
		"""

		py_z = self.py_z
		py = self.py
		pz = self.pz

		ny,nz = py_z.shape

		py = np.tile(py.reshape((-1,1)),(1,nz)) 
		# pz = np.tile(pz.reshape((1,-1)),(ny,1)) 

		sums = np.sum(py_z * (np.log(py_z) - np.log(py)),axis = 0)

		self.contrib = sums*pz

		# dont know if higher is better or the opposite.


class sIB(IB):
	
	# Sequential information bottleneck is one of 'hard' clustering presented in paper.
	# It can be extanded to soft clustering.
	def __init__(self,dim_hidden = 2,data_type='discrete',beta = 0.1, tol = 10**-15,max_iter = 500, seed=None,check_result=False,):
		super().__init__(self,dim_hidden,data_type,beta, tol,max_iter, seed,check_result)
		#self.initialize_ib = initialize_ib
		self.iib = None

	def run_IB(self,X,Y):
		super().fit(X,Y)
		self.iib = 

	def initialize_z(self):
		# The partition is 'hard'
		self.cluster = np.random.randint(dim_hidden,size=self.dim_x)

	def fit(self,X,Y):

		self.calculate_pxy(X,Y)
		self.initialize_z()

		x_supp = np.unique(X)

		# Done = False
		# while not Done:
		# 	Done = True
		# 	for x in x_supp : 
				

