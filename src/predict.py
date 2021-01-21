from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import numpy as np

import config

model = load_model(os.path.join(config.models_path, "my_model.h5"))
model.summary()

#Training Data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(config.final_train_dir,
                                                    target_size = (84, 84), 
                                                    batch_size = 100,
                                                    class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  fill_mode = 'nearest')

test_generator = test_datagen.flow_from_directory(config.test_dir,
                                                  target_size = (84, 84),
                                                  batch_size = 1,
                                                  class_mode = None,    #Only data, no labels
                                                  shuffle = False)      #Keep data in same order as labels

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              steps_per_epoch = 600,
                              epochs = 1,
                              verbose = 1)

#Save the model
model.save(os.path.join(config.models_path, "final_trained_model.h5"))

#model = load_model(os.path.join(config.models_path, "final_trained_model.h5"))
#model.compile(loss='categorical_crossentropy',
#              optimizer=RMSprop(lr=0.01),
#              metrics=['acc'])

predictions = model.predict_generator(test_generator, steps = 10000, verbose = 1)
predict_classes = np.argmax(predictions, axis=1).tolist()

#CSV operations
data = pd.read_csv(os.path.join(config.input_path, 'test.csv'))

data['label'] = predict_classes

data.to_csv(os.path.join(config.input_path, "submission_inception.csv"), columns = ['id', 'label'], index = False)