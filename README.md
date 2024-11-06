# GNSS-FL

GNSS-FL can...


## jammedagents.m



## gnss-fl_continuallearning.py

it runs over different INR, Monte Carlo, time instants
The hierarchy is :
	for each INR value
		for each MC realization
			for each time
				estimate the J. location


if time vector is > 1 but 'aggregated_over_time' is flagged, aggregated output data is computed both over MC simulations and time.

It is possible to obtain an aggregation effect over different random agents position by generating data for a static jammer over obs time > 1. Then by processing those data with aggregate_over_time=1 and optionally Nmc = 1;