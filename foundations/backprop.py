import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        z = np.dot(w, x) + b
        y_pred = 1/(1+np.exp(-z))
        dl = 1
        dy_pred = (y_pred - y_true)*dl
        dz = y_pred*(1-y_pred)*dy_pred
        dw = x*dz
        db = dz

        return (np.round(dw, 5), np.round(db, 5))
