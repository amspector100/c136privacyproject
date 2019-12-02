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


warnings.filterwarnings("ignore")
import plotnine
from plotnine import *
warnings.resetwarnings()


# True location: the true location of the player, or 'mean' if this
# represents a mean over all players. Priv is a flag for whether
# the mechanism is differentially private

# Prior and loc dist are not implemented yet...
ID_COLUMNS = ['trueloc', 'n_players', 'alpha', 'prop', 'eps', 'mech']
FEATURE_COLUMNS = ['util', 'ent', 'util_se']
COLS = ID_COLUMNS + FEATURE_COLUMNS


def get_path(loc_type, dist, num_loc, seed, samples, n_players, alpha, prop, eps):
	""" Could use this for caching - will do that later """
	base_path =  f'data/v1/{loc_type}_dist{dist}_numloc{num_loc}/seed{seed}_samples{samples}'
	base_path += f'_n_players{n_players}_alpha{alpha}_prop{prop}_eps{eps}'
	return base_path

def compare_location_mechs(loc_type, dist = 'uniform',
						   num_loc = 10, n_players = 9, 
						   alpha = 0, prop = 0, eps = 0.1,
						   seed = 136, 	samples_per_batch = 1000):
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
		locations = np.arange(0, num_loc-1, 1)/((num_loc-1)/2) - 1
		locations = np.append(locations, [1], axis = 0)
	elif loc_type == 'skewed':
		locations = np.arange(0, num_loc, 1)/num_loc - 1
		locations = locations.astype('float32')**2
	else:
		raise ValueError(f"loc_type must be one of uniform or skewed, not {loc_type}")


	path = f'data/v1/{loc_type}_dist{dist}_numloc{num_loc}/seed{seed}_samples{samples_per_batch}'


	# Process inputs and create data path... this is a bit complex sorry
	if n_players is None:
		n_players = np.arange(
			1, num_loc + int(num_loc/10 + 1), int(num_loc/10)
		)
		path += '_curven_players'
	else:
		path += f'_n_players{n_players}'
		n_players = [n_players]
	if alpha is None:
		low = -3
		high = 0
		alpha = np.logspace(low, high, base = 10, num = 3*(high - low) + 1)
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
		eps = np.logspace(-5, 2, base = 10, num = 8)
		path += f'_curveeps'
	else:
		path += f'_eps{eps}'
		eps = [eps]

	output_path = path + '_util.csv'
	selection_path = path + '_selection.csv'
	locs_path = path + '_locs.csv'

	# Create directories
	dirname = os.path.dirname(output_path)
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	# All arguments... plus batch args (which are constant for now)
	# We can convert to a list bc if this is big enough to be a problem,
	# the program won't run anyway
	all_args = list(itertools.product(n_players, alpha, prop, eps))
	batch = 1

	# Initialize output
	output = pd.DataFrame(columns = COLS)
	selection_cols = ID_COLUMNS + ['selection_prop']
	selection_output = pd.DataFrame(columns = selection_cols)


	# Iterate through and record values
	for i, arg in enumerate(all_args):

		# Report progress
		print(f'At input {i} of {len(all_args)}')

		n_players0 = int(arg[0]) # Num players
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
			seed = seed, dist = dist
		)
		utils_priv, ents_priv, selections_priv, _, _ = mechsim_private.run_simulation(
			proportion = prop0, alpha = alpha0,
			batch = batch, samples_per_batch = samples_per_batch, 
			eps = eps0, seed = seed, dist = dist
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
				locs[j, 0, 0], n_players0, alpha0, prop0, eps0, 'median',
				player_means[j, 0], ents[j, 0, 0], player_ses[j, 0]
			]
			to_add = pd.Series(to_add, index = COLS)

			# Private results
			priv_to_add = [
				locs[j, 0, 0], n_players0, alpha0, prop0, eps0, 'chen12',
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
		ent_mean = ents.mean()
		social_util_priv, social_util_priv_se = pull_mean_se(utils_priv)
		ent_mean_priv = ents_priv.mean()

		# Add the mean to the df
		to_add = pd.Series(
			['mean', n_players0, alpha0, prop0, eps0, 'median',
			  social_util, ent_mean, social_util_se],
			index = COLS
		)

		priv_to_add = pd.Series(
			['mean', n_players0, alpha0, prop0, eps0, 'chen12',
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
					[l, n_players0, alpha0, prop0, eps0, 'median',p],
					index = selection_cols,
				), ignore_index = True
			)
		# Again for privacy
		for l, p in zip(unique_locs_priv, selection_counts_priv):
			selection_output = selection_output.append(
				pd.Series(
					[l, n_players0, alpha0, prop0, eps0, 'chen12',p],
					index = selection_cols,
				), ignore_index = True
			)

		output.to_csv(output_path, index = False)
		selection_output.to_csv(selection_path, index = False)
	np.savetxt(locs_path, locs[:, 0, 0])

	return path


def plot_location_data(path):
	"""
	:param path: The path to the data. Will automatically
	parse everything else from the path itself. 
	"""

	# util/selection path
	util_path = path + '_util.csv'
	selection_path = path + '_selection.csv'
	locs_path = path + '_locs.csv'

	# Parse path
	params = {}
	params['num_loc'] = path.split('numloc')[-1].split('/')[0]
	for param in ['n_players', 'alpha', 'samples', 'prop', 'eps', 'seed', 'dist']:
		path_part = path.split(param)[-1].split('_')[0]
		if path_part == '':
			params[param] = None
		else:
			try:
				params[param] = float(path_part)
			except:
				params[param] = path_part

	# Create output path
	base_output_path = path.replace('data', 'figures') + '/'
	dirname = os.path.dirname(base_output_path)
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	# Plots of average utility and entropy
	util_data = pd.read_csv(util_path)
	curve_params = [p for p in params if params[p] is None]
	if len(curve_params) == 1:
		curve_param = curve_params[0]
	elif len(curve_params) > 1:
		raise ValueError("Cannot plot curves with more than 1 curve param")
	else:
		curve_param = None

	# Add SEs properly
	util_data['yerr'] = 1.96*util_data['util_se']
	mean_data = util_data.loc[util_data['trueloc'] == 'mean']
	dist = params['dist']
	if curve_param is not None:

		mean_data[curve_param] = np.around(mean_data[curve_param], 5)

		# Plot for utility
		title = f'Average Utilities for 1D Location Mechanisms ({dist} Dist)'
		g1 = (
			ggplot(
				mean_data, aes(
					x = curve_param, y = 'util', fill = 'mech', color = 'mech',
				)
			) 
			+ geom_line()
			+ geom_point()
			+ geom_errorbar(aes(ymin='util-yerr', ymax = 'util+yerr'), width = 0.0001)
			+ labs(title = title, y = 'Utilities')
			+ theme(panel_grid_minor = element_blank())
		)
		if curve_param == 'eps' or curve_param == 'alpha':
			g1 += scale_x_log10()
		g1.save(base_output_path + 'util.JPG')

		# Epsilon
		title = f'Average Entropies for 1D Location Mechanisms ({dist} Dist)'
		g2 = (
			ggplot(
				mean_data, aes(
					x = curve_param, y = 'ent', fill = 'mech', color = 'mech',
				)
			) 
			+ geom_line()
			+ geom_point()
			+ labs(title = title, y = 'Average Normalized Entropy of Strategies')
			+ theme(panel_grid_minor = element_blank())
		)
		if curve_param == 'eps' or curve_param == 'alpha':
			g2 += scale_x_log10()
		g2.save(base_output_path + 'ent.JPG')
	else:
		# Else, just do some bar plots
		title = f'Average Utilities for 1D Location Mechanisms ({dist} Dist)'
		g1 = (
			ggplot(
				mean_data, aes(
					x = 'mech', y = 'util', fill = 'mech', color = 'mech',
				)
			) 
			+ geom_col(position = 'dodge')
			+ geom_errorbar(aes(ymin='util-yerr', ymax = 'util+yerr'), width = 0.5)
			+ labs(title = title, y = 'Utilities')
			+ theme(panel_grid_minor = element_blank())
		)
		g1.save(base_output_path + 'util.JPG')

		title = f'Average Entropies for 1D Location Mechanisms ({dist} Dist)'
		g1 = (
			ggplot(
				mean_data, aes(
					x = 'mech', y = 'ent', fill = 'mech', color = 'mech',
				)
			) 
			+ geom_col()
			+ labs(title = title, y = 'Average Normalized Entropy of Strategies')
			+ theme(panel_grid_minor = element_blank())
		)
		g1.save(base_output_path + 'ent.JPG')

	# Scatter plots by location
	title = f'Average Utilities for 1D Location Mechs, by Location ({dist} Dist)'
	non_mean_data = util_data.loc[util_data['trueloc'] != 'mean']
	non_mean_data['trueloc'] = non_mean_data['trueloc'].astype(float)
	if curve_param is not None:
		non_mean_data[curve_param] = np.around(non_mean_data[curve_param], 5)

	g3 = (ggplot(
			non_mean_data, aes(
				x = 'trueloc', y = 'util', fill = 'mech', color = 'mech',
			)
		) 
		+ geom_point()
		#+ geom_errorbar(aes(ymin='util-yerr', ymax = 'util+yerr'), width = 0.0001)
		+ labs(title = title, y = 'Utilities')
		+ theme(panel_grid_minor = element_blank())
	)
	if curve_param is not None:
		g3 += facet_wrap('~'+curve_param)

	g3.save(base_output_path + 'util_per_loc.JPG')

	title = f'Average Entropies for 1D Location Mechs, by Location ({dist} Dist)'
	g4 = (ggplot(
			non_mean_data, aes(
				x = 'trueloc', y = 'ent', fill = 'mech', color = 'mech',
			)
		) 
		+ geom_point()
		+ labs(title = title, y = 'Entropies')
		+ theme(panel_grid_minor = element_blank())
	)
	if curve_param is not None:
		g4 += facet_wrap('~'+curve_param)

	g4.save(base_output_path + 'ent_per_loc.JPG')

	# Plot selections and locations
	selections = pd.read_csv(selection_path)
	g5 = (ggplot(
		selections, aes(
			x='trueloc', fill = 'mech'
			)
		)
		+ geom_density(aes(weight = 'selection_prop'), alpha = 0.5)
	)
	if curve_param is not None:
		g5 += facet_wrap('~'+curve_param, scales = 'free')

	print(g5)
	g5.save(base_output_path + 'selections.JPG')

if __name__ == '__main__':

	main()