import numpy as np
print(np.sum([0.5, 1.5]))

print(np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32))
print(np.sum([[0, 1], [0, 5]]))

print(np.sum([[2, 1], [2, 5]], axis=1))

print(np.sum([[0, 1], [0, 5]], axis=1))

print(np.sum([[0, 1], [np.nan, 5]], where=[False, True], axis=1))
