import random
import numpy as np

width = 20
height = 20
nodesNumber = 20

xs = np.random.randint(width, size=nodesNumber)
ys = np.random.randint(height, size=nodesNumber)

np.column_stack((xs, ys))

print(xs)
print(ys)
print(np)