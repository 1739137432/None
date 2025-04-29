import numpy as np

arr = np.array([[5,1],[3, 2], [3, 1], [5, 0]])
print(arr)
arr = arr.tolist()
arr.sort(key=lambda x: (x[0], x[1]))
arr=np.array(arr)
print(arr)
