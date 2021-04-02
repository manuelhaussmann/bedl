import torch as th
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math


def Phi(X):
    return 0.5 * (1 + th.erf(X/math.sqrt(2)))


