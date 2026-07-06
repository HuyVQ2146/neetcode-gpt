import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        
        X = np.dot(X, weights)
        return X.round(5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        
        n = model_prediction.shape[0]
        mse = np.sum((model_prediction - ground_truth) ** 2) / n
        return round(float(mse), 5)
