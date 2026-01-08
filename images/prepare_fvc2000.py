from sklearn.model_selection import train_test_split
import numpy as np
import os

DIR = "dataset_FVC2000_DB4_B/dataset/np_data"

data = np.load(os.path.join(DIR, 'img_train.npy'))
label = np.load(os.path.join(DIR, 'label_train.npy'))

train_data, val_data, train_labels, val_labels = train_test_split(data, label, test_size=0.2, random_state=42)

# save arrays

np.save(os.path.join(DIR, 'training_data.npy'), train_data)
np.save(os.path.join(DIR, 'validation_data.npy'), val_data)
np.save(os.path.join(DIR, 'training_labels.npy'), train_labels)
np.save(os.path.join(DIR, 'validation_labels.npy'), val_labels)