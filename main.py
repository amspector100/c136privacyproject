
import os
import sys
import warnings

import numpy as np
import pandas as pd

from privacyproject import discrete, plotting



def main():

	# Step 1: uniform locations
	path = plotting.compare_location_mechs(
		loc_type = 'skewed',
		dist = 'skewed',
		num_loc = 51,
		n_players = 10,
		alpha = None,
		samples_per_batch = 10
	)
	plotting.plot_location_data(path)



if __name__ == '__main__':

	# Possibly profile
	if '--profile' in sys.argv:

		# Profile
		import cProfile
		cProfile.run('main()', '.profile')

		# Analyze
		import pstats
		p = pstats.Stats('.profile')
		p.strip_dirs().sort_stats('cumulative').print_stats(50)

	else:

		main()