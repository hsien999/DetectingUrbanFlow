import math
from numba import njit, float32


@njit([float32(float32, float32)], cache=True)
def add(a, b):
    return a + b


@njit([float32(float32, float32)], cache=True)
def sub(a, b):
    return a - b


@njit([float32(float32, float32)], cache=True)
def mul(a, b):
    return a * b


@njit([float32(float32, float32)], cache=True)
def div(a, b):
    return a / b


@njit([float32(float32)], cache=True)
def power(a):
    return a ** 2


@njit([float32(float32)], cache=True)
def sqrt(a):
    return math.sqrt(a)


@njit([float32(float32)], cache=True)
def log(a):
    return math.log(a)


@njit([float32(float32, float32, float32, float32, float32, float32)], cache=True)
def __expression_1(v1, v2, v3, v4, v5, v6):
    return v1 + v2 + v3 + v4 - v5 - v6


@njit([float32(float32, float32, float32, float32)], cache=True)
def __expression_2(v1, v2, v3, v4):
    return v1 + v2 - v3 - v4


@njit(cache=True)
def euclidean_distance(point1, point2):
    """
    return the Euclidean distance between point1 and point2
    """
    return sqrt(add(power(sub(point1[0], point2[0])), power(sub(point1[1], point2[1]))))


@njit([float32(float32, float32, float32, float32)], cache=True)
def bernoulli_lambda(N_O_r, N_D_r, N_O, N_D):
    """
    return the Bernoulli-based log-likelihood ratio test statistic
    """
    item_1 = mul(N_O_r, log(div(N_O_r, add(N_O_r, N_D_r))))
    item_2 = mul(N_D_r, log(div(N_D_r, add(N_O_r, N_D_r))))
    item_3 = mul(sub(N_O, N_O_r), log(div(sub(N_O, N_O_r), __expression_2(N_O, N_D, N_O_r, N_D_r))))
    item_4 = mul(sub(N_D, N_D_r), log(div(sub(N_D, N_D_r), __expression_2(N_O, N_D, N_O_r, N_D_r))))
    item_5 = mul(N_O, log(div(N_O, add(N_O, N_D))))
    item_6 = mul(N_D, log(div(N_D, add(N_O, N_D))))
    return __expression_1(item_1, item_2, item_3, item_4, item_5, item_6)
