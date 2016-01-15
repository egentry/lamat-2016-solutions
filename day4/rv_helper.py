
import numpy as np 
import scipy

def time_to_theta(t, t_0, P, e):
	"""
	Converts time into true anomaly (theta in our conventions)

	This requires numerical root finding (see scipy.optimize.newton)

	Parameters
	----------
	t : float
		Current time
	t_0 : float
		reference time
	P : float
		Orbital period
	e : float
		Orbital eccentricity

	Returns
	-------
	theta : float
		Mean anomaly
	
	Notes
	-----
	The units of `t`, `t_0` and `P` don't matter, 
	*as long as you use the same units for each!*
	"""

	# first solve for eccentric anomaly (eta)

	phase = 2*np.pi*(t-t_0)/P %(2*np.pi)

	phase = np.array(phase, ndmin=1) # otherwise iterating below fails
	etas = np.empty_like(phase)
	for i, p in enumerate(phase):

		def _solve_this(eta):
			return eta - e*np.sin(eta) - p

		etas[i] = scipy.optimize.brentq(_solve_this, 0, 2*np.pi+.00001)

	# now solve for true anomaly
	theta = 2 * np.arctan(((1+e)/(1-e))**.5 * np.tan(etas/2.))

	return theta

