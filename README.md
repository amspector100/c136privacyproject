# Structure

- Class that picks the median given $n$ reports
- Class that does simulations, with agents 

# To run tests

go to your terminal and run:
``python3 -m unittest``
or 
``python -m unittest``

# To run experiments

Edit the kwargs in main.py and then run
``python3 main.py``
If it's taking a while, you can run
``python3 main.py --profile``
to see which function calls are slowing it down.


# Todo

1. Some graphs of simulations, and maybe 2D location stuff
2. Prove:
	a. That the exponential sampling dist is best response
	b. That the exponential sampling dist is a corr equilibria
3. Think about:
	a. What Bayesian interp means given this particular strat
