#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:19:13 2017

@author: shubhi
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values,y_values)

#error
error=np.mean((y_values-body_reg.predict(x_values))**2)
print ('%2f' %error)

#visualize results
plt.scatter(x_values,y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()