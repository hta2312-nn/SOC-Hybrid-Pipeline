import numpy as np, os

DATA_DIR   = r"C:\new\malware_project\ember_data\ember2018"
N_FEATURES = 2351

y = np.memmap(os.path.join(DATA_DIR, "y_train.dat"),
              dtype=np.float32, mode='r', shape=(800000,))
X = np.memmap(os.path.join(DATA_DIR, "X_train.dat"),
              dtype=np.float32, mode='r', shape=(800000, N_FEATURES))

labeled = y != -1
print(f"Labeled  : {labeled.sum():,}")
print(f"Malware  : {(y[labeled]==1).sum():,}")
print(f"Benign   : {(y[labeled]==0).sum():,}")
print(f"X shape  : {X.shape}")
print("OK - sẵn sàng train")