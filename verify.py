import numpy as np
# Load the training data you just created
xtrain = np.load("X_train.npy")
ytrain = np.load("y_train.npy")

print(f"I have {xtrain.shape[0]} images, and each has {xtrain.shape[1]} features.")
print(f"I have {len(np.unique(ytrain))} different classes (objects).")