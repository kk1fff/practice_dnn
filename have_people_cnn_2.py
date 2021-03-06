from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dropout, Dense, Activation
from keras.optimizers import SGD
import numpy as np

NB_EPOCH = 1000
BATCH_SIZE = 32

data = np.load('human_with_label.npz')
print("Data Loaded")

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
print("Training/Testing sets are built")

def MiniBatchGenerator(batch_size, train, label):
    buf_x = []
    buf_y = []
    while True:
        for x, y in zip(train, label):
            buf_x.append(x)
            buf_y.append(y)
            if len(buf_x) >= 1:
                yield np.array(buf_x), np.array(buf_y)
                buf_x = []
                buf_y = []
    # if len(buf_x) > 0:
    #     yield np.array(buf_x), np.array(buf_y)
        
model = Sequential()
model.add(Convolution2D(64, 20, 20,
                        subsample=(1, 1),
                        border_mode='same',
                        input_shape=(100, 100, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 12, 12,
                        subsample=(1, 1),
                        border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(180))
model.add(Activation("relu"))
model.add(Dense(180))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
print(train.shape)

# Train in mini batch.
print("Training:")
model.fit_generator(MiniBatchGenerator(BATCH_SIZE, train, label_train), BATCH_SIZE, NB_EPOCH)
# for e in range(NB_EPOCH):
#     print("Epoch: {}".format(e))
#     for train, label in MiniBatchGenerator(BATCH_SIZE, train, label_train):
#         model.train_on_batch(train, label)

score = model.evaluate(test, label_test, batch_size=16)
print("Score:")
print(score)

with open("human_model.json", "w") as o:
    o.write(model.to_json())

model.save('human_model_weights.h5')
