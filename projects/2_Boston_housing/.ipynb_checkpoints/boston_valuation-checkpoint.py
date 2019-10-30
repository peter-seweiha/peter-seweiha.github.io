from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd


# Gather and prepare data
boston_dataset = load_boston()
data = pd.DataFrame(data= boston_dataset.data , columns= boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis = 1)

log_prices = np.log(boston_dataset.target)
target= pd.DataFrame(data= log_prices, columns= ['PRICE'])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

property_stats = features.mean().values.reshape(1, 11)

# Create the Linear Regression model
regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

# Calculate MSE & RMSE
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river = False,
                    high_confidence = True):
    
    """ Estimate the price of a property in Boston
    
    Keyword arguments:
    nr_rooms -- number of rooms in the property
    students_per_classroom -- number of students per teacher in the class room for the school in the area
    next_to_river -- True if the property is next to the river, False otherwise
    high_confidence -- True for 95% prediction interval, False for a 68% interval
    
    """
    
    # Error msgs
    if nr_rooms < 1 or students_per_classroom < 1:
        print('no of rooms and students_per_classroom should be more than zero. Please review your entries')
        return
    
    
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
        
    
    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]
    
    # Calc Range
    if high_confidence:
        upper_bound = log_estimate + 2* RMSE
        lower_bound = log_estimate - 2* RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
        
    # Converting to $ and adjusting for inflation and rounded to nearest thousand
    def convert(log_value):
        scale_factor = 27.514150943  # factor due to inflation from the 70s
        return round((np.e **log_value )* 1000 *scale_factor, -3)
    
    
    print (f'Estimated property value: $ {convert(log_estimate)}')
    print (f'At {interval}% the valuation range: ${convert(lower_bound)} - {convert(upper_bound)}')
    
    


