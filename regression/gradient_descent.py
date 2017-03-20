import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_rss_partial(known_values, feature_matrix, weights):
    return -np.dot(np.transpose(feature_matrix), known_values-np.dot(feature_matrix, weights))


def get_rss(y, features, weights):
    return ((y-np.dot(features, weights))**2).sum()


# returns polynomial model for inputs based on degree and weight vector
# initialized to 0
def gen_predicted(x, deg):
    df = pd.DataFrame(x, columns=['x'])
    for i in range(deg+1):
        df['h'+str(i)] = df['x']**i
    return df.drop(['x'], axis=1).as_matrix(), np.zeros(deg+1)


def add_to_plot(weights, label='', start=1, end=4, color=''):
    x_plot = np.arange(start, end, 0.1)
    y_plot = 0
    for index,val in enumerate(weights):
        y_plot += val*x_plot**index
    plt.plot(x_plot,y_plot,color, label=label)
    if label != '':
        plt.legend()
        

def fixed(n, *args):
    return n
        
 
# if there is no step function defined, step size is fixed
def gradient_descent(y, features, weights, step_size, tolerance, verbose=False, step_function=fixed, **step_params):
    y_copy = y.copy()
    features_copy = features.copy()
    weights_copy = weights.copy()
    n=step_size
    i=1
    while True:
        partial = get_rss_partial(y_copy, features_copy, weights_copy)
        n = step_function(n, i, y_copy, features_copy, weights_copy, step_params)
        weights_copy += n*partial
        if np.isnan(np.max(weights_copy)):
            raise RuntimeError('Weights are too large.  May not converge.')
        if np.linalg.norm(partial)<tolerance:
            break
        i+=1
        if verbose and i%50000==0:
            print("loop:", i, ' | n:', n, ' | weights:', weights_copy)
    return weights_copy
