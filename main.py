
import os
import sys
import warnings

import numpy as np
import pandas as pd

from privacyproject import discrete, plotting
from privacyproject.plotting import compare_location_mechs



def main():

	# Step 1: uniform locations
	compare_location_mechs(
		loc_type = 'uniform',
		num_loc = 20,
		eps = None
	)



if __name__ == '__main__':

	main()