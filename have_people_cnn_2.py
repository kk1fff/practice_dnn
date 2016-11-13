from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dropout, Dense, Activation
from keras.optimizers import SGD
import numpy as np

data = np.load('testimgs/human_with_label.npz')
train = data['imgs']/255.
train = np.reshape(train, (-1, 100, 100, 3))
labels = data['labels']

indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0] * 0.2)
test_idx, training_idx = indices[:valid_cnt], \
                         indices[valid_cnt:]

test, train = train[test_idx, :], \
              train[training_idx, :]
label_test, label_train = labels[test_idx, :], \
                          labels[training_idx, :]
model = Sequential()
model.add(Convolution2D(32, 12, 12,
                        subsample=(1, 1),
                        border_mode='same',
                        input_shape=(100, 100, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 6, 6,
                        subsample=(1, 1),
                        border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(90))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
print(train.shape)
model.fit(train, label_train, batch_size=32, nb_epoch=10)
score = model.evaluate(test, label_test, batch_size=16)
print("Score:")
print(score)

with open("human_model.json", "w") as o:
    o.write(model.to_json())

model.save('human_model_weights.h5')
