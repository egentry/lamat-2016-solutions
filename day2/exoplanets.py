try:
	# import block for Python 3
	from urllib.request import urlopen, quote

except ImportError:
	# if that fails, fall back to Python 2 import block
	from urllib2 import urlopen, quote

import numpy as np

filename = "planets_and_stars.dat"

def download_data(filename=filename):
	"""
	Downloads exoplanet and host star information to `filename`

	Data is downloaded through the NASA Exoplanet Archive API.
	Creates a file in CSV format.

	To load this file into python, call `parse_data` (see below)

	Parameters
	----------
	filename : str, optional
		filename of created data file

	Side effects
	------------
		Creates new file `filename`, overwriting existing file if necessary

	Returns
	-------
	None

	"""

	url = "http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets"

	# select which columns to download
	url += "&select=pl_hostname,pl_letter," \
	             + "pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2," \
	             + "pl_bmassj,pl_bmassjerr1,pl_bmassjerr2," \
	             + "st_lum"
	
	# filter out rows with missing data
	url += "&where=pl_hostname     IS NOT NULL" \
		+    " and pl_letter       IS NOT NULL" \
		+    " and pl_orbsmax      IS NOT NULL" \
		+    " and pl_orbsmaxerr1  IS NOT NULL" \
		+    " and pl_orbsmaxerr2  IS NOT NULL" \
		+    " and pl_bmassj       IS NOT NULL" \
		+    " and pl_bmassjerr1   IS NOT NULL" \
		+    " and pl_bmassjerr2   IS NOT NULL" \
		+    " and st_lum          IS NOT NULL" 


	# convert into the valid formatting for a URL
	url = quote(url, safe="&:/?")

	url_request = urlopen(url)
	data = url_request.read().decode()

	with open(filename, mode="w") as f:
		f.write(data)


def parse_data(filename=filename):
	"""
	Parse data from `filename` into a numpy structured array

	Parameters
	------
	filename : str, optional
		Location of data file
		Expects data file is in format generated by `download_data()`

	Returns
	-------
	data : numpy.ndarray (structured)
		columns: 
			pl_hostname : bytestring
				Name of stellar host of planet
			pl_letter : bytestring
				Letter specifying which planet within system

			pl_orbsmax : float
				orbital semi-major axis of planet (in units of AU)
			pl_orbsmaxerr1 : float:
				upper uncertainty in semi-major axis (in units of AU)
			pl_orbsmaxerr2 : float:
				lower uncertainty in semi-major axis (in units of AU)

			pl_bmassj : float
				planet mass (in units of Jupiter masses)
			pl_bmassjerr1 : float:
				upper uncertainty in planet mass (in Jupiter masses)
			pl_bmassjerr2 : float:
				lower uncertainty in planet mass (in Jupiter masses

			st_lum : float
				log_10(star's luminosity / solar luminosity)

	Notes
 	-----
 	How *err1 and *err2 work:
 		We think the true value for pl_orbsmax could be anywhere between
 		pl_orbsmax - abs(pl_orbsmaxerr2) and pl_orbsmax + pl_orbsmaxerr1

 		(and similarly for planet mass)

	"""

	data = np.genfromtxt(filename,
		delimiter=",",
		dtype=None,
		names = True)

	return data
