"""
Thoughts:
1. Meta simulator class?
2. Use itertools/something so that
any of the inputs can be a list
3. Create a pandas DF of outputs,
with things like entropy,
utility, selections, etc
"""

from .import discrete

import os
import sys
import warnings

import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



# True location: the true location of the player, or 'mean' if this
# represents a mean over all players. Priv is a flag for whether
# the mechanism is differentially private

# Prior and loc dist are not implemented yet...
ID_COLUMNS = ['trueloc', 'n_players', 'alpha', 'prop', 'eps', 'priv']
FEATURE_COLUMNS = ['util', 'ent', 'util_se']
COLS = ID_COLUMNS + FEATURE_COLUMNS

def compare_location_mechs(loc_type, num_loc = 10, n_players = 9, 
						   alpha = 1, prop = 0, eps = 0.1,
						   seed = 136):
	"""
	:param loc_type: one of 'uniform' or 'skewed'
	:param num_loc: Number of locations
	:param n_players: The number of players in the game (int). If none,
	will analyze between 1 and 5 * num_loc 
	:param alpha: The level at which individuals care about privacy.
	If None, will analyze between 0 and 5
	:param prop: The proportion of truthful players. If None, will 
	analyze several values between 0 and 1
	:param eps: The differential privacy parameter epsilon: if None, 
	will analyze values on a log scale between 1e-5 and 100
	"""

	# Create location
	if loc_type == 'uniform':
		locations = np.arange(0, num_loc, 1)/(num_loc/2) - 1
	elif loc_type == 'skewed':
		locations = np.arange(0, num_loc, 1)/num_loc - 1
		locations = locations.astype('float32')**2
	else:
		raise ValueError(f"loc_type must be one of uniform or skewed, not {loc_type}")


	path = f'data/v1/{loc_type}_numloc{num_loc}/seed{seed}'


	# Process inputs and create data path... this is a bit complex sorry
	if n_players is None:
		n_players = np.arange(
			1, 5*num_loc + int(num_loc/10 + 1), int(num_loc/10)
		)
		path += '_curven_players'
	else:
		path += f'_n_players{n_players}'
		n_players = [n_players]
	if alpha is None:
		alpha = np.arange(0, 11, 1)/10
		path += '_curvealpha'
	else:
		path += f'_alpha{alpha}'
		alpha = [alpha]
	if prop is None:
		prop = np.arange(0, 6, 1)/5
		path += '_curveprop'
	else:
		path += f'_prop{prop}'
		prop = [prop]
	if eps is None:
		eps = np.logspace(-5, 2, base = 10, num = 16)
		path += f'_curveps'
	else:
		path += f'_eps{eps}'
		eps = [eps]

	output_path = path + '_util.csv'
	selection_path = path + '_selection.csv'

	# Create directories
	dirname = os.path.dirname(output_path)
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	# All arguments... plus batch args (which are constant for now)
	all_args = itertools.product(n_players, alpha, prop, eps)
	batch = 1
	samples_per_batch = 1000

	# Initialize output
	output = pd.DataFrame(columns = COLS)
	selection_cols = ID_COLUMNS + ['selection_prop']
	selection_output = pd.DataFrame(columns = selection_cols)


	# Iterate through and record values
	for arg in all_args:
		n_players0 = arg[0] # Num players
		alpha0 = arg[1] # alpha
		prop0 = arg[2] # proportion of truthful players
		eps0 = arg[3] # Epsilon in diff privacy

		# Initialize mechanisms
		mechsim = discrete.MedianLocationSimulator(
			n_players = n_players0, locations = locations
		)
		mechsim_private = discrete.MedianLocationSimulator(
			n_players = n_players0, 
			locations = locations,
			private = True
		)

		# Run simulation
		utils, ents, selections, inds, locs = mechsim.run_simulation(
			proportion = prop0, alpha = alpha0,
			batch = batch, samples_per_batch = samples_per_batch, 
			seed = seed
		)
		utils_priv, ents_priv, selections_priv, _, _ = mechsim_private.run_simulation(
			proportion = prop0, alpha = alpha0,
			batch = batch, samples_per_batch = samples_per_batch, 
			eps = eps0, seed = seed
		)

		# Add utilities for each location - recall that first dim of
		# all these arrays is the true location of players
		player_means = utils.mean(axis = -1)
		player_ses = utils.std(axis = -1)/np.sqrt(samples_per_batch)
		player_means_priv = utils_priv.mean(axis = -1)
		player_ses_priv = utils_priv.std(axis = -1)/np.sqrt(samples_per_batch)

		# Loop through and add to dataframe
		for j in range(n_players0):

			# Non-private results
			to_add = [
				locs[j, 0, 0], n_players0, alpha0, prop0, eps0, False,
				player_means[j, 0], ents[j, 0, 0], player_ses[j, 0]
			]
			to_add = pd.Series(to_add, index = COLS)

			# Private results
			priv_to_add = [
				locs[j, 0, 0], n_players0, alpha0, prop0, eps0, True,
				player_means_priv[j, 0], ents_priv[j, 0, 0], player_ses_priv[j, 0]
			]
			priv_to_add = pd.Series(priv_to_add, index = COLS)

			# Output
			output = output.append(to_add, ignore_index = True)
			output = output.append(priv_to_add, ignore_index = True)

		# Compare social utility
		def pull_mean_se(x):

			mu = x.mean(axis = 0).mean(axis = -1)[0]
			se = x.mean(axis = 0).std(axis = -1)[0]/np.sqrt(samples_per_batch)

			return mu, se

		# Regular social utility
		social_util, social_util_se = pull_mean_se(utils)
		ent_mean = ents.mean(axis = 0)[0,0]
		social_util_priv, social_util_priv_se = pull_mean_se(utils_priv)
		ent_mean_priv = ents_priv.mean(axis = 0)[0, 0]

		# Add the mean to the df
		to_add = pd.Series(
			['mean', n_players0, alpha0, prop0, eps0, False,
			  social_util, ent_mean, social_util_se],
			index = COLS
		)

		priv_to_add = pd.Series(
			['mean', n_players0, alpha0, prop0, eps0, True,
			social_util_priv, ent_mean_priv, social_util_priv_se],
			index = COLS
		)
		output = output.append(to_add, ignore_index = True)
		output = output.append(priv_to_add, ignore_index = True)

		# Finally, add selections
		unique_locs, selection_counts = np.unique(
			selections, return_counts = True
		)
		selection_counts = selection_counts / samples_per_batch
		# And again for private version
		unique_locs_priv, selection_counts_priv = np.unique(
			selections_priv, return_counts = True
		)
		selection_counts_priv = selection_counts_priv / samples_per_batch

		# Loop through locations again to add to output
		for l, p in zip(unique_locs, selection_counts):
			selection_output = selection_output.append(
				pd.Series(
					[l, n_players0, alpha0, prop0, eps0, False,p],
					index = selection_cols,
				), ignore_index = True
			)
		# Again for privacy
		for l, p in zip(unique_locs_priv, selection_counts_priv):
			selection_output = selection_output.append(
				pd.Series(
					[l, n_players0, alpha0, prop0, eps0, True,p],
					index = selection_cols,
				), ignore_index = True
			)

		output.to_csv(output_path, index = False)
		selection_output.to_csv(selection_path, index = False)



if __name__ == '__main__':

	main()