import numpy as np



class MedianLocation():
	"""
	:param n_players: Number of players
	:param locations: A list of n numbers, representing
	locations on 1D plane
	"""

	def __init__(self, n_players, locations):

		self.n_players = n_players
		self.locations = locations

		if not isinstance(self.locations, np.ndarray):
			self.locations = np.array(self.locations)


	def select(self, reports):
		"""
		:param reports: If there are N players and L locations,
		a length L array of reports from each player. Each report
		should be an index (from 1 to n) indexing into the list of
		locations.
		"""

		if not isinstance(reports, np.ndarray):
			reports = np.array(reports).astype('int32')
		else:
			reports = reports.astype('int32')

		return np.median(self.locations[reports])


class MedianLocationSimulator():
	"""
	:param n_players: number of players
	:param locations: A list of n numbers, representing
	locations on a 1D plane

	For now, this will use discrete uniform priors
	"""


	def __init__(self, n_players, locations):

		# Initialization
		self.n_players = np.array(n_players)
		self.locations = np.array(locations)
		self.L = locations.shape[0]
		self.mechanism = MedianLocation(n_players, locations)

	def sample_locations(self, batch = 100):

		selections = np.random.randnint(low = 0, high = self.L,
										size = (self.n_players, batch))

		locations = np.array([self.locations[selection] for selection in selections])
		
		return selections




if __name__ == '__main__':

	pass