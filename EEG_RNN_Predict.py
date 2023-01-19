import pandas
import numpy as np
from sklearn import preprocessing
from numpy import var
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


SEQ_LEN = 120
FUTURE_SAMPLE_PREDICT = 24 #five seconds multiplied by the sampling frequency

target_data = 'EEG'
target_EEG_Channels = ['1']


data = sio.loadmat('normalized_EEG.mat')

# Maybe use lower init-ranges.
contents = data['EEG']
#print(contents[1,2])
a = np.transpose(contents[:,:])
#we transpose to get into a form more similar to a tutorial
input_df = pandas.DataFrame(a)
input_df.columns =  [ '1', '2', '3', '4', '5' , '6' , '7', '8', '9', '10', '11', '12', '13','14', '15', '16']
input_df.index.names = ['Time Stamp']
#note channel numbers become indexed at zero
#print(main_df)
#print(df)

#create a new column in df called future aimed at the future values. of a certain channel, in the future will change this to other ECoG channels.
#ain_df['future'] = main_df['1'].shift(-FUTURE_SAMPLE_PREDICT)
df_targets = input_df[target_EEG_Channels].shift(-FUTURE_SAMPLE_PREDICT)
#print(main_df)
df_targets.columns = ['future']
#print(df_targets)
#print(input_df)
#plt.plot(main_df)
#plt.show()


#now convert pandas data frames to numpy arrays so it can be input into nerual net.
x_data = input_df.values[0:-FUTURE_SAMPLE_PREDICT]
y_data = df_targets.values[:-FUTURE_SAMPLE_PREDICT]

#number of observations
num_data = len(x_data)
#split data into training and test sets
train_split = 0.9
num_train = int(train_split * num_data)
num_test = num_data -num_train
#splitting the data into training and test
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)
# output signals size
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
(len(y_train) + len(y_test))
# number of input signals
num_x_signals = x_data.shape[1]
#print(num_x_signals)
#number of output signals
num_y_signals = y_data.shape[1]
#print(type(y_data))
#print("Shape:", y_data.shape[1])

def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array fo r the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index. MUST CHANGE THIS TO SCALED DATA
            x_batch[i] = x_train[idx:idx+sequence_length]
            y_batch[i] = y_train[idx:idx+sequence_length]

        yield (x_batch, y_batch)
        #batch generator variables initialization
batch_size = 60
sequence_length = 1000
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)
print(x_batch.shape)
print(y_batch.shape)

def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


def plot_comparison(start_idx, length=1000, train=True):
    """
    Plot the predicted and true output-signals.

    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = x_train
        y_true = y_train
    else:
        # Use test-data.
        x = x_test
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    #plots the input EEG values
    #    plt.plot(x, label = 'input values')
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.


    # For each output-signal.

    # Get the output-signal predicted by the model.


    signal_pred = y_pred[0,:]

    # Get the true output-signal from the data-set.
    signal_true = y_true[:, 0]


    # Make the plotting-canvas bigger.
    plt.figure(figsize=(15,5))

    # Plot and compare the two signals.
    plt.plot(signal_true, label='true')
    plt.plot(signal_pred, label='pred')

    # Plot grey box for warmup-period.
    p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

    # Plot labels etc.
    plt.ylabel(1)
    plt.legend()
    plt.show()


#This gives us a random batch of 256 sequences, each sequence having 2000 observations, and each observation having 16 input-signals and 1 output-signals.

#plot one of the 16 input signals
batch = 0   # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
#plt.plot(seq)
#plt.ylabel('xx')

#plot one of the output signals
seq = y_batch[batch, :, signal]
#plt.plot(seq)

# data preperation
validation_data = (np.expand_dims(x_test, axis=0),
                   np.expand_dims(y_test, axis=0))

#callback functions
path_checkpoint = '24_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./24_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


                #create the Recurrent Neural Network Architecture

model = Sequential()

model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

model.add(Dense(num_y_signals, activation='sigmoid'))

warmup_steps = 50

optimizer = RMSprop(lr=1e-3)

model.compile(loss=loss_mse_warmup, optimizer=optimizer)


model.summary()

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


#silence the below code in order to predict the function

#model.fit_generator(generator=generator,
                #    epochs=1,
                    #steps_per_epoch=100,
                    #validation_data=validation_data,
                    #callbacks=callbacks)

plot_comparison(start_idx=2000, length=1000, train=True)


def standardize(df):
    x = main_df.values
    scaler = preprocessing.StandardScaler()
    standardize = scaler.fit_transform(x)
    df = pandas.DataFrame(standardize)
    print(df)
