from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D

# we load the images and perform some preprocessing techniques.
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

# we define the batch and target size here and creates a training and validation object
batchSize= 32
targetSize=(24, 24)
training_batch= generator('data/training', shuffle=True, batch_size=batchSize, target_size=targetSize)
validation_batch= generator('data/validation', shuffle=True, batch_size=batchSize, target_size=targetSize)
# let's calculate the number of steps per epoch and validation step
StepsPerEpoch= len(training_batch.classes) // batchSize
ValidationSteps = len(validation_batch.classes) // batchSize

model = Sequential([
    # Convolutional 2D Layer
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    # Max pooling layer
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),


# we added a dropout layer here.randomly turn neurons on and off during training to prevent overfitting and improve convergence
    Dropout(0.25),
# we flatten the output to 1D vector since there's too many dimensions.
    Flatten(),
# Let's add fully connected layer to get all relevant data. we used relu activation function.
    Dense(128, activation='relu'),
# We adds one additional dropout layer for the sake of convergence)
    Dropout(0.5),
# Let's add a final dense layer with 2 neurons and a softmac activation function.
    Dense(2, activation='softmax')
])
# Let's compile the mode. we use adam optimizer for optimization and categorical cross entropy function as a loss function.
# we used accuracy metrics to perform evaluation during training.
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Let's train our model
model.fit_generator(training_batch, validation_data=validation_batch, epochs=15, steps_per_epoch=StepsPerEpoch, validation_steps=ValidationSteps)
# Let's Save Our Model
model.save('models/cnnSleep.h5', overwxrite=True)