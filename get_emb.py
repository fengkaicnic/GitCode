from keras.models import load_model
from keras.models import Model
import pdb

model = load_model('model_weight.h5')
pdb.set_trace()

flat_layer = Model(model.input, outputs=model.get_layer('flatten_1').output)

flat_out = falt_layer.predict()

