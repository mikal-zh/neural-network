##
## EPITECH PROJECT, 2025
## G-CNA-500-PAR-5-1-neuralnetwork-4
## File description:
## maths_math
##
import math

def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def ReLu(x: float):
    if x < 0:
        return 0
    return x

def ReLu_derivative(x: float):
    if x > 0:
        return 1
    return 0

def leakyReLu(x: float):
    if x > 0:
        return x
    return 0.01 * x

def leakyReLu_derivative(x: float):
    if x > 0:
        return 1
    return 0.01


def softMax(outputs):
    maximum = max(outputs)
    results = [math.exp(out - maximum) for out in outputs]
    total = sum(results)
    return [res / total for res in results]
