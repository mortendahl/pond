from tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor, AnalyticTensor
from nn import Sequential, Dense, Sigmoid, Diff, DataLoader

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


x = DataLoader(x_train, NativeTensor)
y = DataLoader(y_train, NativeTensor)
model.initialize()
model.fit(x, y, Diff(), epochs=2000, learning_rate=.1)
print(model.predict(x))
print("Native training done.")


# x = DataLoader(x_train, PublicEncodedTensor)
# y = DataLoader(y_train, PublicEncodedTensor)
# model.initialize()
# model.fit(x, y, Diff(), epochs=2000, learning_rate=.1)
# print(model.predict(x))
# print("Encoded training done.")


x = DataLoader(x_train, PrivateEncodedTensor)
y = DataLoader(y_train, PrivateEncodedTensor)
model.initialize()
model.fit(x, y, Diff(), epochs=2000, learning_rate=.1)
print(model.predict(x))
print("Private training done.")

# AnalyticTensor.reset()
# x = AnalyticTensor(x_train)
# y = AnalyticTensor(y_train)
# model.initialize()
# model.fit(x, y, Diff(), epochs=1, learning_rate=.1)
# ops = AnalyticTensor.store()
# for op in ops:
#     print(op)