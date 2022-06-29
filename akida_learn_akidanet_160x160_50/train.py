import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging, math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Activation, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam

from keras import Model
from conversion import convert_akida_to_tf_lite, save_saved_model
from cnn2snn import quantize, check_model_compatibility, convert
from akida_models.layer_blocks import dense_block 
from akida_models import akidanet_imagenet

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Sync HRV Logger files to Edge Impulse')
parser.add_argument('--x-file', type=str, required=False)
parser.add_argument('--y-file', type=str, required=False)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--validation-set-size', type=float, required=True)
parser.add_argument('--input-shape', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=False)

args = parser.parse_args()

# for --x-file, --y-file, --out-directory use the defaults (used by Edge Impulse), if not passed in
x_file = args.x_file if args.x_file else '/home/X_train_features.npy'
y_file = args.y_file if args.y_file else '/home/y_train.npy'
out_directory = args.out_directory if args.out_directory else '/home'

if not os.path.exists(x_file):
    print('--x-file argument', x_file, 'does not exist', flush=True)
    exit(1)
if not os.path.exists(y_file):
    print('--y-file argument', y_file, 'does not exist', flush=True)
    exit(1)
if not os.path.exists(out_directory):
    os.mkdir(out_directory)

X = np.load(x_file)
Y = np.load(y_file)[:,0]

classes = np.max(Y)

# get the shape of the input, and reshape the features
MODEL_INPUT_SHAPE = tuple([ int(x) for x in list(filter(None, args.input_shape.replace('(', '').replace(')', '').split(','))) ])
X = X.reshape(tuple([X.shape[0] ]) + MODEL_INPUT_SHAPE)

# convert Y to a categorical vector
Y = tf.keras.utils.to_categorical(Y - 1, classes)

# split in train/validate set and convert into TF Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.validation_set_size, random_state=1)

# split in train/validate set and convert into TF Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.validation_set_size, random_state=1)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []

# NOTE: Define model architecture here, see Brainchip's MetaTF documentation to learn how 
# to create model architectures compatible with Akida:
#   https://doc.brainchipinc.com/examples/cnn2snn/plot_0_cnn_flow.html
# or, select from Brainchip's set of transfer learning architectures:
#   https://doc.brainchipinc.com/zoo_performances.html

# Load in a transfer learning models
# Here we are loading both the floating point and QAT base layers in order to compare performance
FLOAT_WEIGHTS_PATH = './transfer-learning-weights/akidanet_imagenet_160_alpha_50_iq8_wq4_aq4.h5'

base_model_float = akidanet_imagenet(input_shape=MODEL_INPUT_SHAPE,
                               classes=classes,
                               alpha=0.5,
                               include_top=False,
                               pooling='avg',
                               weight_quantization=4,
                               activ_quantization=4,
                               input_weight_quantization=8)
base_model_float.load_weights(FLOAT_WEIGHTS_PATH, by_name=True)

# build the float model
base_model_float.trainable = False
x = base_model_float.output
x = Flatten(name='flatten')(x)
x = dense_block(x,
                units=128,
                name='fc1',
                add_batchnorm=True,
                add_activation=True)
x = Dropout(0.2, name='dropout_1')(x)
x = dense_block(x,
                units=classes,
                name='predictions',
                add_batchnorm=False,
                add_activation=False)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((classes,), name='reshape')(x)
model_float = Model(base_model_float.input, x, name='')


lr = tf.keras.optimizers.schedules.PolynomialDecay(
    args.learning_rate,
    args.epochs,
    end_learning_rate=args.learning_rate / 10.0,
    power=1.0,
    cycle=False,
    name=None,
)
model_float.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# this controls the learning rate
opt = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 32
train_dataset_batch = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset_batch = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# train the neural network
model_float.fit(train_dataset_batch, epochs=args.epochs, validation_data=validation_dataset_batch, verbose=2, callbacks=callbacks)

print('')
print('Initial training done.', flush=True)
print('')
print('Finetuning model with quantization aware training...', flush=True)
print('')

# Quantize weights and activation to 4 bits, first layer weights to 8 bits
model_quantized = quantize(model_float, 4, 4, 8)

# How many epochs we will fine tune the model with QAT
FINE_TUNE_EPOCHS = 30
model_quantized.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000045),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model_quantized.fit(train_dataset_batch,
                epochs=FINE_TUNE_EPOCHS,
                verbose=2,
                validation_data=validation_dataset_batch,
                callbacks=callbacks,
                class_weight=None
            )
 
# Save the models to disk
save_saved_model(model_quantized, out_directory)

print('')
print('Testing model conversion to Akida')
print('')

check_model_compatibility(model_quantized)

model_akida = convert(model_quantized)
model_akida.summary()
# Save .fbz (later retrieved by deploy block when bundling model artifacts for export)
model_quantized.save(os.path.join(out_directory, "trained.fbz"))

# Create tflite files (NOTE: Used only for metrics in EI studio) - converted directly from float/quantized keras models
convert_akida_to_tf_lite(model_float, model_quantized, out_directory, validation_dataset, MODEL_INPUT_SHAPE,
    'model.tflite', 'model_quantized_int8_io.tflite', False)
