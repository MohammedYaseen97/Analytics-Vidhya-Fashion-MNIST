from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import config

#Pre trained model and local weights file
pre_trained_model = InceptionV3(input_shape=(84, 84, 3),
                                include_top=False,
                                weights = None)

pre_trained_model.load_weights(config.local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

#Customising it to our data
last_layer = pre_trained_model.get_layer('mixed7')

last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(10, activation = 'softmax')(x)

model = Model(pre_trained_model.input, x)

#Save the model
model.save(os.path.join(config.models_path, "my_model.h5"))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['acc'])

#Training and validation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   fill_mode = 'nearest')
train_generator = train_datagen.flow_from_directory(config.new_train_dir, 
                                                    target_size = (84, 84), 
                                                    batch_size = 100,
                                                    class_mode = 'categorical')

val_datagen = ImageDataGenerator(rescale = 1./255,
                                 fill_mode = 'nearest')
val_generator = val_datagen.flow_from_directory(config.val_dir, 
                                                target_size = (84, 84), 
                                                batch_size = 100, 
                                                class_mode = 'categorical')

history = model.fit_generator(train_generator,
                              steps_per_epoch = 480,
                              epochs = 1,
                              validation_data = val_generator,
                              validation_steps = 120,
                              verbose = 1)
