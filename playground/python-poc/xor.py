from tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor
from nn import Sequential, Dense, Sigmoid, Diff

import numpy as np


x_train = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]).reshape(4,2)

y_train = np.array([
    0,
    0,
    1,
    1
]).reshape(4,1)


model = Sequential([
    Dense(4, 2),
    Sigmoid(),
    Dense(1, 4),
    Sigmoid()
])


x = NativeTensor(x_train)
y = NativeTensor(y_train)
model.initialize()
model.fit(x, y, Diff(), epochs=2000, learning_rate=.1)
print(model.predict(x))
print("Native training done.")


x = PublicEncodedTensor(x_train)
y = PublicEncodedTensor(y_train)
model.initialize()
model.fit(x, y, Diff(), epochs=2000, learning_rate=.1)
print(model.predict(x))
print("Encoded training done.")


x = PrivateEncodedTensor(x_train)
y = PrivateEncodedTensor(y_train)
model.initialize()
model.fit(x, y, Diff(), epochs=2000, learning_rate=.1)
print(model.predict(x))
print("Private training done.")