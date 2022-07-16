import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

'''c = np.array(43)
a = np.array([1,2,3,4,5])
b = np.array([[1,2,3],[4,5,6]])
d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(d)
print(d.ndim)

arr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr[0,1,2])

arr = np.array([[1,2,3],[4,5,6]])
print(arr[1, 1:3])
print(arr[0:2, 1:3])

arr = np.array([1,2,3,4,5])
print(arr[1:5])
print(arr[::2])
print(arr[-3:-1])

arr = np.array([1.1,2.3,3.5])
new_arr = arr.astype('U')

print(new_arr)
print(new_arr.dtype)

arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr = arr.reshape(4,3)
new_arr = arr.reshape(2,3,2)

print(new_arr)

arr = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

for x in np.nditer(arr[:, :, ::2]):
    print(x)

arr = np.array([[11,20,31],[8,4,5]])
for idx, x in np.ndenumerate(arr):
    print(idx, x)

arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])

arr = np.stack((arr1, arr2), axis=1)
arr_2 = np.hstack((arr1, arr2))
arr_3 = np.vstack((arr1, arr2))
arr_4 = np.dstack((arr1, arr2))

print(arr_4)

arr = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]])
newarr = np.hsplit(arr, 3)
print(newarr)

arr = np.array([1,22,3,24,15,3,6,7,81,9,10])
y = np.where(arr == 3)
x = np.sort(arr)
print(x)

arr = np.array([12,30, 37, 31,40,50,60])
filter_arr = arr > 35
new_arr = arr[filter_arr]
print(filter_arr)
print(new_arr)

x = random.randint(1000, size=(3,5))
print(x)

p = random.choice([3,4,6,2,8], size=(3,5))
print(p) 

y = np.array([1,2,34,5,6,22])
print(random.permutation(y))

sns.distplot(y, hist = False)
plt.show()

sns.distplot(random.normal(loc=50, scale=5, size=1000),hist=False)
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False)
sns.distplot(random.exponential(size=1000), hist=False)


plt.show()'''

