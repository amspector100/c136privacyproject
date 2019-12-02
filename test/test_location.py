import numpy as np
import unittest
from .context import privacyproject

from privacyproject import discrete


class TestBincount(unittest.TestCase):
	""" Tests utility function """

	def test_correctness(self):

		minlength = 5
		# Shape is: (2, 4, 3)
		x = np.array([[
			[1, 2, 3],
			[1, 1, 1], 
			[1, 2, 1],
			[0, 1, 0]],[
			[1, 1, 1],
			[1, 1, 1],
			[1, 1, 1],
			[1, 1, 1],
		]])

		out1 = discrete.batched_bincount(x, minlength = 5, axis = 0)
		out2 = discrete.batched_bincount(x, minlength = 5, axis = 1)

		# Test for correct shapes
		self.assertEqual(
			out1.shape, (4, 3, 5), 
			msg = 'Batched bincount returns wrong-shaped object'
		)
		self.assertEqual(
			out2.shape, (2, 3, 5), 
			msg = 'Batched bincount returns wrong-shaped object'
		)

		# Test for correct counts
		np.testing.assert_array_almost_equal(
			out1[0, 0], np.array([0, 2, 0, 0, 0]),
			err_msg = 'Incorrect counts for batched_bincount'
		)
		np.testing.assert_array_almost_equal(
			out2[1, 2], np.array([0, 4, 0, 0, 0]),
			err_msg = 'Incorrect counts for batched_bincount'
		)



class TestLocation(unittest.TestCase):
	""" Tests discrete simulation functions """

	def test_location(self):
		""" Tests correct location selection""" 

		# Params
		n_players = 5
		locations = [-10, -5, 0, 1, 2, 3, 4, 5, 6]
		reports = [0, 1, 3, 6, 8]

		# Init
		mechanism = discrete.MedianLocation(n_players = n_players, locations = locations)

		# Test
		output = mechanism.select(reports)

		self.assertEqual(
			output, 1,
			msg = 'Location mechanism does not correctly select median'
		)

		# Test batched version
		reports2 = [[0, 1, 3, 6, 8], [0, 1, 4, 6, 8]]
		output2 = mechanism.select(reports2)#tolist()


	def test_diff_priv_location(self):


		# Params
		n_players = 5
		locations = [-10, -5, 0, 1, 2, 3, 4, 5, 6]
		reports = [0, 1, 3, 6, 8]

		# Init
		np.random.seed(110)
		mechanism = discrete.DiffPrivLoc(n_players = n_players, locations = locations)
		output = mechanism.select(reports, eps = 100)
		self.assertEqual(
			output, 1,
			msg = 'Diff privacy location mechanism does not correctly select median'
		)

		# Test batched version
		reports2 = np.array([
			[[0, 1, 3, 7, 5], [0, 1, 4, 0, 8]],
			[[2, 1, 4, 8, 8], [1, 1, 1, 3, 5]],
			[[2, 1, 5, 8, 8], [1, 1, 1, 3, 5]]
		])
		reports2 = np.transpose(reports2, axes = [2, 0, 1])
		np.random.seed(110)
		output2 = mechanism.select(reports2, eps = 100)#.tolist()
		np.testing.assert_array_almost_equal(
			output2, np.array([[1, -5], [2, -5], [3, -5]]),
			err_msg = 'Diff privacy location mech selects wrong median for batched inputs'
		)


	def test_sampling(self):

		n_players = 5
		locations = [-1, -.5, 0, .1, .2, .3, .4, .5, .6]
		L = len(locations)
		reports = [0, 1, 3, 6, 8]

		batch = 100
		mechsim = discrete.MedianLocationSimulator(n_players, locations)
		selections, locs = mechsim.sample_locations(batch = batch)

		self.assertEqual(
			selections.shape[0], n_players,
			msg = 'LocationSim samples incorrectly (wrong num players)'
		)
		self.assertEqual(
			selections.shape[1], batch,
			msg = 'LocationSim samples incorrectly (wrong num batches)'
		)
		self.assertEqual(
			locations[selections[0, 0]], locs[0, 0],
			msg = 'Index/location relationship is off after sampling'
		)

		# Test the same thing but for skewed sampling
		np.random.seed(110)
		batch = 1000
		selections2, locs2 = mechsim.sample_locations(batch = batch, dist = 'skewed')

		self.assertEqual(
			selections2.shape[0], n_players,
			msg = 'LocationSim samples incorrectly for "skewed" dist (wrong num players)'
		)
		self.assertEqual(
			selections2.shape[1], batch,
			msg = 'LocationSim samples incorrectly for "skewed" dist (wrong num batches)'
		)
		self.assertEqual(
			locations[selections2[0, 0]], locs2[0, 0],
			msg = 'Index/location relationship is off after sampling for "skewed" dist'
		)

		# Check to make sure dist is properly skewed
		counts2 = np.bincount(selections2.flatten(), minlength = L).tolist()
		self.assertEqual(
			counts2, sorted(counts2),
			msg = 'Skewed sampling is not properly skewed'
		)



	def test_calc_expectations(self):

		# Create n_players, locations, etc
		n_players = 5
		locations = [-1, 0, .1, .2, .3, .4, .5, .6]
		mechsim = discrete.MedianLocationSimulator(n_players, locations)

		# Calc expectation
		utils, ses = mechsim.calc_expectations(batch = 10000)
		self.assertTrue(
			np.max(utils[0]) < -1,
			msg = 'Incorrectly calculates utilities'
		)


	def test_simulation(self):

		# Create n_players, location, etc
		n_players = 5
		locations = [-1, 0, .1, .2, .3, .4, .5, .6]
		mechsim = discrete.MedianLocationSimulator(n_players, locations)
		priv_mechsim = discrete.MedianLocationSimulator(n_players, locations, private = True)

		# First, test that seeding works
		_,_,_,inds,locs = mechsim.run_simulation(
			batch = 1, samples_per_batch = 10000, seed = 110
		)

		_,_,_,inds2,locs2 = mechsim.run_simulation(
			batch = 1, samples_per_batch = 1, seed = 110
		)

		np.testing.assert_array_almost_equal(
			inds, inds2, err_msg = 'Inconsistent sampling from simulation method'
		)
		np.testing.assert_array_almost_equal(
			locs, locs2, err_msg = 'Inconsistent sampling from simulation method'
		)

		# ...and again for private mechsim
		_,_,_,inds,locs = priv_mechsim.run_simulation(
			batch = 1, samples_per_batch = 10000, seed = 110
		)

		_,_,_,inds2,locs2 = priv_mechsim.run_simulation(
			batch = 1, samples_per_batch = 1, seed = 110
		)

		np.testing.assert_array_almost_equal(
			inds, inds2, err_msg = 'Inconsistent sampling from simulation method'
		)
		np.testing.assert_array_almost_equal(
			locs, locs2, err_msg = 'Inconsistent sampling from simulation method'
		)

		# And test that sampling is different with skewed dist
		_,_,_,inds3,locs3 = priv_mechsim.run_simulation(
			batch = 1000, samples_per_batch = 1, seed = 110, dist = 'skewed'
		)
		counts3 = np.bincount(inds3.flatten(), minlength = len(locations)).tolist()
		self.assertEqual(
			counts3, sorted(counts3),
			msg = 'Skewed sampling does not pass through to run_simulation method'
		)


if __name__ == '__main__':
	unittest.main()