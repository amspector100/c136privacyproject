import numpy as np
import unittest
from .context import privacyproject

from privacyproject import discrete

class TestLocation(unittest.TestCase):
	""" Tests some various utility functions """


	def test_location(self):
		""" Tests correct location selection""" 

		# Params
		print('helloooooo')
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


if __name__ == '__main__':
	unittest.main()