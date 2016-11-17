import sys
import skimage.io
import glob
from keras.models import model_from_json

with open("human_model.json", "r") as json_file:
    model = model_from_json(json_file.read())

model.load_weights('human_model_weights.h5')

for fn in glob.glob(sys.argv[1]):
    skimage.io.imread(fn, mode="RGB")
    img_arr = scipy.misc.imresize(img_arr, [100, 100])
    r = model.predict(img_arr)

    if r[0][0] > 0.5:
        result = "Yes"
    else:
        result = "No"

    print("{}: {}".format(fn, result))
