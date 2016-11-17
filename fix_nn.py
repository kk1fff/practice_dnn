from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dropout, Dense, Activation
from keras.optimizers import SGD
import numpy as np

NB_EPOCH = 1000
BATCH_SIZE = 32

data = np.load('fixer_train.npz')
print("Data Loaded")

train = data['trainings']/255.
labels = data['results']/255.

# could be a 3-channel image, flatten
#train = np.reshape(train, (-1,))
#labels = np.reshape(labels, (-1,))

OUT_DIM = labels.shape[1]

indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0] * 0.2)
test_idx, training_idx = indices[:valid_cnt], \
                         indices[valid_cnt:]

test, train = train[test_idx, :], \
              train[training_idx, :]
label_test, label_train = labels[test_idx, :], \
                          labels[training_idx, :]
print("Training/Testing sets are built")

def MiniBatchGenerator(batch_size, train, label):
    buf_x = []
    buf_y = []
    while True:
        for x, y in zip(train, label):
            buf_x.append(x)
            buf_y.append(y)
            if len(buf_x) >= batch_size:
                yield np.array(buf_x), np.array(buf_y)
                buf_x = []
                buf_y = []
        
model = Sequential()

model.add(Dense(input_shape=train.shape[1:], output_dim=200))
model.add(Activation("relu"))
model.add(Dense(150))
model.add(Activation("relu"))
model.add(Dense(150))
model.add(Activation("relu"))
model.add(Dense(90))
model.add(Activation("relu"))
model.add(Dense(80))
model.add(Activation("relu"))
model.add(Dense(70))
model.add(Activation("relu"))
model.add(Dense(output_dim=OUT_DIM))
model.add(Activation("tanh"))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['mean_squared_error'])
print(train.shape)

# Train in mini batch.
print("Training:")
model.fit_generator(MiniBatchGenerator(BATCH_SIZE, train, label_train),
                    BATCH_SIZE,
                    NB_EPOCH)

score = model.evaluate(test, label_test, batch_size=16)
print("Score:")
print(score)

with open("picture_fix.json", "w") as o:
    o.write(model.to_json())

model.save('picture_fix.h5')
