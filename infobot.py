import numpy as np  # Tested with 1.8.0
from scipy.special import logsumexp  # Tested with 0.13.0
import torch


class InformationBottleneck(object):
	"""
	One dimensional information bottleneck.
	"""
	def __init__(self, n_hidden=2, dim_hidden=2,            # Size of representations
					seed = None,									# Random seed
					use_GPU = False):						# Using GPU with torch

		# self.n_hidden = n_hidden
		self.dim_hidden = dim_hidden

		# if type(seed) == int:
		np.random.seed(seed)# Set for deterministic results
		# torch.manual_seed(seed) # Set for deterministic results

		if torch.cuda.is_available() and use_GPU:
			self.device = torch.device("cuda")
			torch.cuda.set_device(0)
			print("Torch device : " + torch.cuda.get_device_name(0))
			# MYABE BETTER TO USE CUDA_VISIBLE_DEVICES
			# if type(seed) == int:
			#     torch.backends.cudnn.deterministic = True # Set for deterministic results
			#     torch.backends.cudnn.benchmark = False # Set for deterministic results
		else:
			print("torch CPU uses")
			self.device = torch.device("cpu")


	def event_from_sample(self, x):
		"""Transform data into event format.
		For each variable, for each possible value of dim_visible it could take (an event),
		we return a boolean matrix of True/False if this event occurred in this sample, x.
		Parameters:
		x: {array-like}, shape = [n_visible]
		Returns:
		x_event: {array-like}, shape = [n_visible * self.dim_visible]
		"""
		x = np.asarray(x)
		n_visible = x.shape[0]

		assert self.n_visible == n_visible, \
			"Incorrect dimensionality for samples to transform."

		return np.ravel(x[:, np.newaxis] == np.tile(np.arange(self.dim_visible), (n_visible, 1)))

	def events_from_samples(self, X):
		"""Transform data into event format. See event_from_sample docstring."""
		n_samples, n_visible = X.shape
		events_to_transform = np.empty((self.n_events, n_samples), dtype=bool)
		for l, x in enumerate(X):
			events_to_transform[:, l] = self.event_from_sample(x)
		return events_to_transform



	def initialize_parameters(self, X):
		"""Set up starting state
		Parameters
		----------
		X : array-like, shape = [n_samples, n_visible]
			The data.
		"""

		self.n_samples, self.n_visible = X.shape
		self.initialize_events(X)
		self.initialize_representation()

	def initialize_events(self, X):
		values_in_data = set(np.unique(X).tolist())-set([self.missing_values])
		self.dim_visible = int(max(values_in_data)) + 1
		if not set(range(self.dim_visible)) == values_in_data:
			print("Warning: Data matrix values should be consecutive integers starting with 0,1,...")
		self.n_events = self.n_visible * self.dim_visible

	def initialize_representation(self):

		self.tc_history = []
		self.tcs = torch.zeros(self.n_hidden,device = self.device,dtype = torch.float)

		log_p_y_given_x_unnorm = -np.log(self.dim_hidden) * (0.5 + np.random.random((self.n_hidden, self.n_samples, self.dim_hidden)))
		log_p_y_given_x_unnorm = torch.tensor(log_p_y_given_x_unnorm,device = self.device)
		#log_p_y_given_x_unnorm = -100.*np.random.randint(0,2,(self.n_hidden, self.n_samples, self.dim_hidden))
		self.p_y_given_x, self.log_z = self.normalize_latent(log_p_y_given_x_unnorm)

	def data_statistics(self, X_event):
		p_x = torch.sum(X_event, dim=1,dtype = torch.float)
		p_x = p_x.view((self.n_visible, self.dim_visible))
		p_x /= torch.sum(p_x, dim=1, keepdim=True)  # With missing values, each x_i may not appear n_samples times
		z = torch.zeros(1,device = self.device)
		ep = torch.tensor(1e-10,device = self.device)
		entropy_x = torch.sum(torch.where(p_x>0., -p_x * torch.log(p_x), z), dim=1)
		entropy_x = torch.where(entropy_x > 0, entropy_x, ep)
		return p_x, entropy_x

	def update_marginals(self, X_event, p_y_given_x):
		self.log_p_y = self.calculate_p_y(p_y_given_x)
		self.log_marg = self.calculate_p_y_xi(X_event, p_y_given_x) - self.log_p_y

	def calculate_p_y(self, p_y_given_x):
		"""Estimate log p(y_j) using a tiny bit of Laplace smoothing to avoid infinities."""
		pseudo_counts = 0.001 + torch.sum(p_y_given_x.float(), dim=1, keepdim=True)
		log_p_y = torch.log(pseudo_counts) - torch.log(torch.sum(pseudo_counts, dim=2, keepdim=True))
		return log_p_y

	def calculate_p_y_xi(self, X_event, p_y_given_x):
		"""Estimate log p(y_j|x_i) using a tiny bit of Laplace smoothing to avoid infinities."""
		pseudo_counts = 0.001 + torch.matmul(X_event.float(), p_y_given_x.float())  # n_hidden, n_events, dim_hidden
		log_marg = torch.log(pseudo_counts) - torch.log(torch.sum(pseudo_counts, dim=2, keepdim=True))
		return log_marg  # May be better to calc log p(x_i|y_j)/p(x_i), as we do in Marg_Corex

	def calculate_mis(self, log_p_y, log_marg):
		"""Return normalized mutual information"""
		vec = torch.exp(log_marg + log_p_y)  # p(y_j|x_i)
		smis = torch.sum(vec * log_marg, dim=2)
		smis = smis.view((self.n_hidden, self.n_visible, self.dim_visible))
		mis = torch.sum(smis * self.p_x, dim=2, keepdim=True)
		return mis/self.entropy_x.view((1, -1, 1))

	def update_alpha(self, mis, tcs):
		t = (self.tmin + self.ttc * torch.abs(tcs)).view((self.n_hidden, 1, 1))
		maxmis = torch.max(mis, dim=0).values
		alphaopt = torch.exp(t * (mis - maxmis))
		self.alpha = (1. - self.lam) * self.alpha.float() + self.lam * alphaopt

	def calculate_latent(self, X_event):
		""""Calculate the probability distribution for hidden factors for each sample."""
		alpha_rep = self.alpha.repeat_interleave(repeats = self.dim_visible, dim=1)
		log_p_y_given_x_unnorm = (1. - self.balance) * self.log_p_y + torch.matmul(X_event.T.float(), alpha_rep*self.log_marg)

		return self.normalize_latent(log_p_y_given_x_unnorm)

	def normalize_latent(self, log_p_y_given_x_unnorm,is_np = False):
		"""Normalize the latent variable distribution
		For each sample in the training set, we estimate a probability distribution
		over y_j, each hidden factor. Here we normalize it. (Eq. 7 in paper.)
		This normalization factor is quite useful as described in upcoming work.
		Parameters
		----------
		Unnormalized distribution of hidden factors for each training sample.
		Returns
		-------
		p_y_given_x : 3D array, shape (n_hidden, n_samples, dim_hidden)
			p(y_j|x^l), the probability distribution over all hidden factors,
			for data samples l = 1...n_samples
		log_z : 2D array, shape (n_hidden, n_samples)
			Point-wise estimate of total correlation explained by each Y_j for each sample,
			used to estimate overall total correlation.
		"""
		if not is_np:
			log_z = torch.logsumexp(log_p_y_given_x_unnorm, dim=2)  # Essential to maintain precision.
			log_z = log_z.view((self.n_hidden, -1, 1))
			res = torch.exp(log_p_y_given_x_unnorm - log_z), log_z
		else:
			log_z = logsumexp(log_p_y_given_x_unnorm, axis=2)  # Essential to maintain precision.
			log_z = log_z.reshape((self.n_hidden, -1, 1))
			res = np.exp(log_p_y_given_x_unnorm - log_z), log_z
		return res

	def update_tc(self, log_z):
		self.tcs = torch.mean(log_z, dim=1).view(-1)
		sum_tcs = torch.sum(self.tcs)
		if self.device == torch.device('cuda'):
			sum_tcs = sum_tcs.cpu()
		self.tc_history.append(sum_tcs.item())

	def sort_and_output(self):
		order = torch.argsort(self.tcs,descending=True)  # Order components from strongest TC to weakest
		self.tcs = self.tcs[order]  # TC for each component
		self.alpha = self.alpha[order]  # Connections between X_i and Y_j
		self.p_y_given_x = self.p_y_given_x[order]  # Probabilistic labels for each sample
		self.log_marg = self.log_marg[order]  # Parameters defining the representation
		self.log_p_y = self.log_p_y[order]  # Parameters defining the representation
		self.log_z = self.log_z[order]  # -log_z can be interpreted as "surprise" for each sample
		if hasattr(self, 'mis'):
			self.mis = self.mis[order]

	def print_verbose(self):
		if self.verbose:
			print(self.tcs)
		if self.verbose > 1:
			print(self.alpha[:,:,0])
			if hasattr(self, "mis"):
				print(self.mis[:,:,0])

	def convergence(self):
		if len(self.tc_history) < 10:
			return False
		dist = -np.mean(self.tc_history[-10:-5]) + np.mean(self.tc_history[-5:])
		return np.abs(dist) < self.eps # Check for convergence. dist is nan for empty arrays, but that's OK

	def new_increm(self,x):
		if x.ndim > 1:
			x = x.reshape(-1)
		x_event = self.event_from_sample(x)
		n_new_samples = x_event.shape[0]
		x_event = torch.tensor(x_event,device=self.device)

		if n_new_samples != self.n_samples:
			print('Bootstrap needed')
			return self

		# representations init
		log_p_y_given_x_unnorm = -np.log(self.dim_hidden) * (0.5 + np.random.random((self.n_hidden, self.n_new_samples, self.dim_hidden)))
		log_p_y_given_x_unnorm = torch.tensor(log_p_y_given_x_unnorm,device = self.device)

		log_p_y_given_x_prev_unnorm = torch.log(torh.tensor(self.p_y_given_x,device  =self.device))

		log_p_y_given_x_unnorm = log_p_y_given_x_unnorm + log_p_y_given_x_prev_unnorm

		self.p_y_given_x, self.log_z = self.normalize_latent(log_p_y_given_x_unnorm)

			
	def TC(self):
		# COMPUTE REAL TC(X). NOT SURE ABOUT THE ESTIMATION ERROR

		p_x_joint = (self.X_event.float().mean(axis=1).reshape(self.n_visible,-1))
		