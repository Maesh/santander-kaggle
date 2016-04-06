"""
Need to work on a class or something to do some
sort of automated data cleaning. Take a data frame,
look at every column, normalize if necessary, break
apart columns with abnormal values.

Begun by Ryan Gooch, Apr, 2016
"""

import pandas as pd
import numpy as np 

class DataEngineer():
	"""
	Engineers Data to one's desire.
	Imports and returns data frame.
	Options to be added to allow specific
	transformations.

	Until that ambitious task is completed,
	I will implement some other functionality
	for now
	"""
	def __init__(self, df, missing=None):
		self.df = df
		
		if missing is not None:
			self.missing = missing


	def value_counter(self,filename='values.csv') :
		"""
		Returns a dataframe containing value counts by variable,
		also writes it to csv.
		"""
		values_list = []
		for col in self.df.columns :
			values_list.append(self.df[col].value_counts())
		values_df = pd.DataFrame(values_list, columns = self.df.columns)
		values_df.to_csv(filename)
		return values_df
