import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gradient_descent_single_step(output, features, weights, step):
    partial = reg.get_rss_partial(output, features, weights)
    return weights - step * partial, partial


def get_curve(output, features, weights, step, iterations):
    """ Calculates the learning curve
    
    Returns:
        array: The output of the magnitude of the gradient descent partial
            over a range of iterations
    """
    weights_temp = weights.copy()
    magnitude = []
    for x in xrange(iterations):
        weights_temp, partial = gradient_descent_single_step(output, features, weights_temp, step)
        magnitude.append(np.linalg.norm(partial))
    return magnitude


def plot_learning_curve(output, features, weights, step_size, iterations):
    """Plot learning curve
    
    Plots a set of learning curve base of a list of step sizes to show 
    how fast they learn
    
    Args:
        step_size (array_like): A list of step sizes.  Use either
            numpy.logspace or numpy.linespace to get create this 
            array.
    """
    for step in step_size:
        learning_curve  = get_curve(output, features, weights, step, iterations)
        plt.plot(np.arange(1, iterations+1, 1), learning_curve)
    plt.title('Learning Curve')
    plt.show()

    
def eval_model(model, data, x, y, start=0, end=12000):
    """Plots a poly regression model against the data
    
    Args:
        data (DataFrame): data
        x (str): name of x column in data
        y (str): name of y column in data
        start (int): Start range for regression model
        end (int): End range for regression model
    """
    plt.plot(data[x], data[y], '.')
    mlg.add_to_plot(model, np.arange(start, end, 1))
    plt.title('Regression Model')
    plt.show()

    
def plot_regression(output, features, weights, data, x, y, step_size, iterations, start=0, end=12000):
    """Plots learning curves and model
    
    Selects the first model in the learning curves
    
    Args:
        data (DataFrame): data
        x (str): name of x column in data
        y (str): name of y column in data
        start (int): Start range for regression model
        end (int): End range for regression model
        step_size (array_like): A list of step sizes.  Use either
            numpy.logspace or numpy.linespace to get create this 
            array.
            
    Returns:
        array: Weights associated with the select regression model
    """
    plot_learning_curve(output,features, weights, step_size, iterations)
    model = weights.copy()
    for x in xrange(iterations):
        model, partial = gradient_descent_single_step(output, features, model, step_size[0])
    eval_model(model, data, x, y, start=start, end=end)
    return model
