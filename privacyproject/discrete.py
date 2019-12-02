import numpy as np
import warnings
import copy
import itertools

# Slightly useful...
from scipy import stats


def unif_to_log(x):
	""" Helper function for profiling""" 
	return -1*np.log(-1*np.log(x))

def batched_bincount(x, minlength, axis = 0):
	""" Batched bincount 
	:param x: Numpy array of integers between 0 and minlength - 1
	:param minlength: Minimum length of output (max val + 1 of x)
	:param axis: Axis to count along. E.g. if x has dim n by m by k,
	and axis = 1, output will be of dimension n by k by minlength
	(counts along m axis)
	"""

	# Permute to put counted-axis last
	num_axes = len(list(x.shape))
	permute_order = [x for x in range(num_axes)]
	permute_order.pop(axis)
	permute_order.append(axis)
	x = np.transpose(x, axes = permute_order)

	# # Create output shape
	output_shape = list(x.shape)[0:-1]
	output_shape.append(minlength)
	output = np.zeros(output_shape)


	# Create indexes to loop along
	iter_axes = [list(range(l)) for l in list(x.shape)[0:-1]]
	products = itertools.product(*iter_axes)

	for item in products:

		# Bincount and assign
		to_add = np.bincount(x[item], minlength = minlength)
		output[item] = to_add

	return output

class MedianLocation():
	"""
	:param n_players: Number of players
	:param locations: A list of n numbers, representing
	locations on 1D plane
	"""

	def __init__(self, n_players, locations, warn = False):

		if not isinstance(n_players, int):
			raise ValueError(f"n_players ({n_players}) must be int, not ({type(n_players)})")

		self.n_players = n_players
		self.locations = locations

		if not isinstance(self.locations, np.ndarray):
			self.locations = np.array(self.locations)
		self.L = self.locations.shape[0]

		if warn:
			if np.max(np.abs(self.locations)) > 1:
				msg = 'Locations are not normalized'
				msg += '(i.e. they do not lie between -1 and 1) -'
				msg += 'this may lead to overflow/numerical errors'
				warnings.warn(msg)

	def preprocess_reports(self, reports):
		""" Pretty self explanatory: preprocesses reports"""

		if not isinstance(reports, np.ndarray):
			reports = np.array(reports).astype('int32')
		else:
			reports = reports.astype('int32')

		# Reports
		if reports.shape[0] != self.n_players:

			# Detect if reports needs to be transposed
			if reports.shape[1] == self.n_players:
				#print("Transposing reports to keep it consistent w/ correct num playes")
				reports = reports.T

			# Else raise valueerror
			else:
				raise ValueError(f'Number of reports {reports.shape[0]} != num players {self.n_players}')

		return reports

	def select(self, reports):
		"""
		:param reports: If there are N players and L locations,
		a length L x (batch dims) array of reports from each player. Each report
		should be an index (from 1 to n) indexing into the list of
		locations.
		"""

		reports = self.preprocess_reports(reports)
		return np.median(self.locations[reports], axis = 0)

	def calc_utils(self, selection, reports, metric = 'L2'):
		"""
		Calculates utility 
		:param selection: The actual chosen location. This is NOT
		an index, it is the ACTUAL location. Should be a scalar.
		:param reports: If there are N players and L locations,
		a length L array of reports from each player. Each report
		should be an index (from 1 to n) indexing into the list of
		locations.
		:param metric: A string indicating which distance/utility
		metric to use (defaults to L2 metric)
		 """

		# Put into numpy array...
		if not isinstance(reports, np.ndarray):
			reports = np.array(reports).astype('int32')
		else:
			reports = reports.astype('int32')

		# Fetch reported locations
		reported_locations = self.locations[reports]

		# Calc utils
		if metric.lower() == 'l2':
			return -1*np.sqrt((reported_locations - selection)**2) + 2
		else:
			raise ValueError('Currently metric {metric} is not supported')


class DiffPrivLoc(MedianLocation):

	def __init__(self, n_players, locations, warn = False):

		super().__init__(n_players, locations, warn = warn) 


	def select(self, reports, eps = 0.1):
		""" Reports as before, eps as in definition of diff privacy """

		# Preprocessing of reports as always
		reports = self.preprocess_reports(reports)

		# Calculate distribution of noise
		noise_dist = np.exp(-1*eps * np.arange(0, self.L+1, 1)/2)
		noise_dist = noise_dist/noise_dist.sum()

		# ...and log probabiltiies
		log_noise_dist = np.log(noise_dist)

		# Use gumbel sampling to get batched samples
		# from log_noise_dist
		gumbel_shape = list(reports.shape)
		gumbel_shape.pop(0)
		batch_shape = copy.copy(gumbel_shape)
		gumbel_shape.append(self.L + 1)
		gumbel_shape.append(self.L + 1)
		gumbel_vals = unif_to_log(np.random.uniform(size = gumbel_shape).astype('float32'))
		noise = np.argmax(log_noise_dist + gumbel_vals, axis = -1)

		# Add to report counts
		report_counts = batched_bincount(
			reports, axis = 0, minlength = self.L+1
		)
		total_counts = report_counts + noise

		# Calculate median (a bit roundabount)
		cum_counts = total_counts.cumsum(axis = -1)

		# Transpose... kind of annoying
		ax_order = [x for x in range(len(list(cum_counts.shape)))]
		num_axes = max(ax_order) + 1
		ax_order.remove(num_axes - 1)
		ax_order.insert(0, num_axes - 1)
		cum_counts = np.transpose(cum_counts, axes = ax_order)

		# Calculate medians!
		cutoffs = cum_counts[-1]/2
		median_inds = np.argmax(cum_counts >= cutoffs, axis = 0)
		medians = self.locations[median_inds]

		return medians


class MedianLocationSimulator():
	"""
	:param n_players: number of players
	:param locations: A list of n numbers, representing
	locations on a 1D plane

	For now, this will use discrete uniform priors
	"""


	def __init__(self, n_players, locations, private = False):

		# Initialization
		self.n_players = n_players
		self.locations = np.array(locations)
		self.L = self.locations.shape[0]
		self.private = private

		if private:
			self.mechanism = DiffPrivLoc(n_players, locations, warn = True)
		else:
			self.mechanism = MedianLocation(n_players, locations, warn = True)

		# DP 
		self.cached_utils = None


	def sample_locations(self, batch = 100, dist = 'uniform'):

		# Each location is equally likely
		if dist == 'uniform':
			selections = np.random.randint(low = 0, high = self.L,
											size = (self.n_players, batch))

		# Or not - skewed to the right
		elif dist == 'skewed':

			# Create skewed probabilities favoring locations to the right
			probs = np.arange(0, self.L, 1) + 1
			probs = probs**2
			probs = probs/probs.sum()
			logits = np.log(probs)
			logits = np.repeat(logits, self.n_players * batch)
			logits = logits.reshape(
				self.L, self.n_players, batch
			)

			# Gumbel noise as usual
			gumbel_values = np.random.gumbel(
				size = (self.L, self.n_players, batch)
			)

			# Take maximums
			selections = np.argmax(logits + gumbel_values, axis = 0)


		locations = np.array([self.locations[selection] for selection in selections])

		return selections, locations

	def calc_expectations(self, batch = 1000, cache = True, dist = 'uniform', **kwargs):
		"""
		Calculates the expected utility of reporting 'index'
		given true location 'true_index' when all other reported 
		locations are drawn from a (discrete) uniform prior via monte-carlo sampling.
		Uses 'batch' samples.

		returns: if there are L locations, an L x L array called utils,
		where utils[i][j] is the expected utility of reporting j given
		true location i.

		"""

		# Sample many locations...
		sample_inds, sample_locs = self.sample_locations(batch = batch, dist = dist)
		
		# Initialize result
		utils = np.zeros((self.L, self.L))

		# Track standard errors to see if we should be
		# sampling more		
		ses = np.zeros((self.L, self.L))

		for index in np.arange(0, self.L, 1):

			# Set the first player's location to the 
			# thing we want to evaluate
			sample_inds[0, :] = index
			selections = self.mechanism.select(reports = sample_inds, **kwargs)

			# Now, calculate utilities
			for true_index, true_loc in enumerate(self.locations):

				dists = -1*np.sqrt((selections - true_loc)**2)
				utils[true_index][index] = dists.mean()
				ses[true_index][index] = dists.std()/np.sqrt(batch)

		# Make them all positive - add maximum pairwise dist
		utils += 2

		if cache:
			if self.cached_utils is not None:
				warnings.warn('Overwriting cached utils')
			self.cached_utils = utils

		return utils, ses


	def run_simulation(self,
					   proportion = 0,
					   alpha = 1,
					   batch = 1000, 
					   samples_per_batch = 1,
					   seed = None,
					   recalc_expecations = False,
					   dist = 'uniform',
					   **kwargs):
		"""
		:param proportion: The proportion of agents who care
		about their privacy.
		:param alpha: How much weight privacy-valuing agents
		put on the privacy (ranges from 0 to infinity)
		:param batch: How many different "true location" values
		to sample
		:param samples_per_batch: How many samples to compute 
		for each batch.
		:param seed: If not None, sets the internal random seed to 
		this value before sampling player locations (so we can 
		analyze similar mechanisms many times).
		:param **kwargs: keyword arguments for selection mech
		(e.g. epsilon if private = True)

		The runtime is linear in samples_per_batch * batch - 
		it's pretty speedy(~3 sec for 1 million samples on a laptop)

		EDIT: this is still linear for the private mechanism,
		but it's about 4x slower (11 sec for 1 million samples).
		"""

		# Possibly calculate utilities
		# Recall this is true location x other locations
		if self.cached_utils is None or recalc_expecations:
			self.calc_expectations(
				batch = batch*samples_per_batch, cache = True, 
				dist = dist, **kwargs
			)

		# Create locations for each player batch times.
		# This is an self.L x batch matrix
		if seed is not None:
			np.random.seed(seed)
		sample_inds, sample_locs = self.sample_locations(batch = batch, dist = dist)

		# Create exponential strategies/distributions for them - 
		# prevent overflows here
		if alpha/np.log(self.L) >= 1e-2:
			strats = np.exp(np.log(self.L)*self.cached_utils/alpha)
			denom = np.expand_dims(strats.sum(axis = 1), axis = -1)
			strats = strats/denom
		else:
			strats = np.eye(self.L)

		# Find the corresponding strategies for each batch
		batch_strats = strats[sample_inds]

		# Find corresponding entropies for later analysis :)
		
		#entropies = -1*(strats * np.log(strats)).sum(axis = 0)
		entropies = stats.entropy(strats.T)/np.log(self.L)
		batch_entropies = entropies[sample_inds]
		batch_entropies = np.expand_dims(batch_entropies, axis = -1)

		# Use the gumbel trick to do batched categorical sampling
		# and create reports based on strategies
		report_shape = (self.n_players, batch, self.L, samples_per_batch)
		# Shape: num_players x batch x self.L x 1
		batch_strats = np.expand_dims(batch_strats, axis = -1)

		# Basically: argmax(log probs + gumbel noise) = categorical sample
		log_probs = np.log(batch_strats)
		noise = np.random.gumbel(size = report_shape)
		false_reports = np.argmax(log_probs + noise, axis = 2)

		# Decide which players are truthful
		truth_mask = np.random.binomial(
			1, proportion, size = (self.n_players, batch, 1)
		)

		# Interpolate with true reports
		true_reports = np.repeat(sample_inds, samples_per_batch)
		true_reports = true_reports.reshape(self.n_players, batch, samples_per_batch)

		# Report mask
		reports = truth_mask * true_reports + (1 - truth_mask) * false_reports
		batch_entropies = (1-truth_mask) * batch_entropies

		# Create selections given reports
		selections = self.mechanism.select(reports, **kwargs)
		selections = np.expand_dims(selections, axis = 0)

		# Calculate utilities
		sample_locs = np.expand_dims(sample_locs, -1)
		utilities = -1*np.sqrt((selections - sample_locs)**2) + 2

		return utilities, batch_entropies, selections, sample_inds, sample_locs

if __name__ == '__main__':

	pass