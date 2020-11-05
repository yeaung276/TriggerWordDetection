from keras.layers import Input, BatchNormalization, GRU, Dropout, Conv1D, Activation
from keras.layers import Dense, TimeDistributed
from keras.models import Model


def create_tr_model(input_shape, name='tr_model' ):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer 
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                   # ReLu activation
    X = Dropout(rate=0.8)(X)                                    # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units=128, return_sequences=True)(X)                # GRU (use 128 units and return the sequences)
    X = Dropout(rate=0.8)(X)                                    # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    # Step 3: Second GRU Layer
    X = GRU(units=128, return_sequences=True)(X)                # GRU (use 128 units and return the sequences)
    X = Dropout(rate=0.8)(X)                                    # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Dropout(rate=0.8)(X)                                    # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1,activation='sigmoid'))(X)       # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X, name=name)
    
    return model  