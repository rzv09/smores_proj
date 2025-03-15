import numpy as np
import matplotlib.pyplot as plt

def plot_preds(prediction, truth):
    plt.figure(figsize=(10, 5))
    plt.plot(truth, label='Actual')
    plt.plot(prediction, label='Predicted')
    plt.legend()
    plt.show()