import os
import pickle
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import concurrent.futures as cf

def resize_dataset(dataset):
    """
        Resizes dataset of MNIST images to (32, 32).

        Parameters:
        -dataset: a numpy array of size [?, 1, 28, 28].
    """        
    args = [dataset[i:i+1000] for i in range(0, len(dataset), 1000)]
    
    def f(chunk):
        return transform.resize(chunk, (chunk.shape[0], 1, 32, 32))

    with cf.ThreadPoolExecutor() as executor:
        res = executor.map(f, args)
    
    res = np.array([*res])
    res = res.reshape(-1, 1, 32, 32)
    return res


def dataloader(X, y, BATCH_SIZE):
    """
        Returns a data generator.

        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t:t+BATCH_SIZE, ...], y[t:t+BATCH_SIZE, ...]
        
def one_hot_encoding(y):
    """
        Performs one-hot-encoding on y.
        
        Parameters:
        - y: ground truth labels.
    """
    N = y.shape[0]
    Z = np.zeros((N, 10))
    Z[np.arange(N), y] = 1
    return Z

def train_val_split(X, y, val=50000):
    """
        Splits X and y into training and validation set.

        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    X_train, X_val = X[:val, :], X[val:, :]
    y_train, y_val = y[:val, :], y[val:, :]

    return X_train, y_train, X_val, y_val

def save_params_to_file(model):
    """
        Saves model parameters to a file.

        Parameters:
        -model: a CNN architecture.
    """
    # Make save_weights/ accessible from every folders.
    terminal_path = ["my_cnn/weights/"]
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("save_params_to_file(): Impossible to find save_weights/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    weights = model.get_params()
    if dirPath == 'save_weights/': # We run the code from demo notebook.
        with open(dirPath + "demo_weights.pkl","wb") as f:
            pickle.dump(weights, f)
    else:
        with open(dirPath + "final_weights.pkl","wb") as f:
            pickle.dump(weights, f)

def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def im2col(X, HF, WF, stride, pad):
    """
        Transforms input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform matrix back to the input image.

        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    
    # Index matrices, necessary to transform our input image into a matrix. 
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
