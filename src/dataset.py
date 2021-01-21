import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile

import config

def create_dataset(train_file, train_dir, output_dir, val=True):
    categories = list(pd.unique(train_file.label))
#    print(categories)
    
    if val:
        new_train_file, val_file = train_test_split(train_file, test_size=0.2, random_state=47, shuffle=True)
        print("Training_size = {}, Validation_size = {}".format(new_train_file.shape, val_file.shape))

        os.mkdir(os.path.join(output_dir, "new_train"))
        os.mkdir(os.path.join(output_dir, "val"))

        for label in categories:
            os.mkdir(os.path.join(output_dir, "new_train", str(label)))
            os.mkdir(os.path.join(output_dir, "val", str(label)))

        for index, row in new_train_file.iterrows():
            copyfile(os.path.join(train_dir, str(row['id'])+'.png'), os.path.join(output_dir, "new_train", str(row['label']), str(row['id'])+'.png'))

        for index, row in val_file.iterrows():
            copyfile(os.path.join(train_dir, str(row['id'])+'.png'), os.path.join(output_dir, "val", str(row['label']), str(row['id'])+'.png'))
            
        return (str(os.path.join(config.input_path, "new_train")), str(os.path.join(config.input_path, "val")))
            
    else:
        os.mkdir(os.path.join(output_dir, "final_train"))

        for label in categories:
            os.mkdir(os.path.join(output_dir, "final_train", str(label)))

        for index, row in train_file.iterrows():
            copyfile(os.path.join(train_dir, str(row['id'])+'.png'), os.path.join(output_dir, "final_train", str(row['label']), str(row['id'])+'.png'))
            
        return str(os.path.join(config.input_path, "final_train"))


train_file = pd.read_csv(os.path.join(config.input_path, "train.csv"))

new_train_dir, val_dir = create_dataset(train_file, config.train_dir, config.input_path, val=False)
print(new_train_dir)
print(val_dir)