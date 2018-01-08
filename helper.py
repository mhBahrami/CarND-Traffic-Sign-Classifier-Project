import csv
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
from sklearn.utils import shuffle
from collections import Counter
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def read_data(file_name):    
    with open(file_name, mode='rb') as f:
        data = pickle.load(f)
        
    return data['features'], data['labels']


def save_data(file_name, X, y):
    print('Save processed data...')
    data = {'features': X, 'labels': y}

    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
    print('Done!\n')   

        
def read_train_data():
    file_name = 'train.p'
    return read_data(file_name)
    
    
def read_valid_data():
    file_name= 'valid.p'
    return read_data(file_name)
    
    
def read_test_data():
    file_name = 'test.p'
    return read_data(file_name)


def save_processed_train_data(X_train_p, y_train_p):
    file_name = 'processed_train.p'
    save_data(file_name, X_train_p, y_train_p)


def save_processed_valid_data(X_valid_p, y_valid_p):
    file_name = 'processed_valid.p'
    save_data(file_name, X_valid_p, y_valid_p)


def save_processed_test_data(X_test_p, y_test_p):
    file_name = 'processed_test.p'
    save_data(file_name, X_test_p, y_test_p)
    

def read_processed_train_data():
    file_name = 'processed_train.p'
    return read_data(file_name)
    
    
def read_processed_valid_data():
    file_name= 'processed_valid.p'
    return read_data(file_name)
    
    
def read_processed_test_data():
    file_name = 'processed_test.p'
    return read_data(file_name)
    

def read_sign_names():
    infile = open('signnames.csv', mode='r')
    reader = csv.DictReader(infile)
    SignNames = {int(row['ClassId']):row['SignName'] for row in reader}
    infile.close()
    
    return SignNames


def on_draw_data_histogram(y_train, y_valid, y_test):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(13, 3)
    ax0, ax1, ax2= axes.flatten()
    n_bins = 43

    # the histogram of the training data
    ax0.hist(y_train, n_bins, normed=1, histtype='bar', alpha=0.75)
    ax0.set_xlabel('ClasId')
    ax0.set_ylabel('Probability')
    ax0.set_title("Training Data Histogram")
    ax0.axis([0, 42, 0, .1])
    ax0.grid(True)

    # the histogram of the testing data
    ax1.hist(y_test, n_bins, normed=1, histtype='bar', alpha=0.75)
    ax1.set_xlabel('ClasId')
    ax1.set_ylabel('Frequency')
    ax1.set_title("Testing Data Histogram")
    ax1.axis([0, 42, 0, .1])
    ax1.grid(True)

    # the histogram of the validation data
    ax2.hist(y_valid, n_bins, normed=1, histtype='bar', alpha=0.75)
    ax2.set_xlabel('ClasId')
    ax2.set_ylabel('Frequency')
    ax2.set_title("Validation Data Histogram")
    ax2.axis([0, 42, 0, .1])
    ax2.grid(True)

    fig.tight_layout()
    plt.show()
    
def normalize_gr(img):
    # It must be a gray scale image
    if (len(img.shape) != 3 or img.shape[2] != 1):
        raise ValueError('The input image shape:{0}\nThe input image must be a gray scale image.'.format(img.shape))
        
    mean = np.mean(img)
    std = np.std(img)
    img_p = np.array(img - mean) / std
    return img_p


def normalize128(img):
    img_p = np.copy(img)
    img_p = (img_p - 128) / 128
    return img_p


def normalize(img):
    img_p = np.asfarray(np.copy(img))
    mean = np.mean(img_p)
    std = np.std(img_p)
    img_p = (img_p - mean) / std
    return img_p


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    result = cv2.LUT(img, table)
    
    return result


def rgb2gray(img):
    img_c = np.copy(img)
    img_g = np.zeros((img.shape[0], img.shape[1], 1))
    img_g[:,:,0] = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    return img_g


def adjust_contrast(img, clip_limit=4, x_tile_size=8, y_tile_size=8):
    tile_grid_size = (int(img.shape[1] // float(x_tile_size)),int(img.shape[0] // float(y_tile_size)))
    transform = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    result = np.zeros((img.shape[0], img.shape[1], 1))
    result[:,:,0] = transform.apply(np.copy(img).astype("uint8"))
    
    return result
    
    
def image_preprocessing(img):
    img_c = np.copy(img)
    img_ga = adjust_gamma(img, gamma=1.1)
    img_gr = rgb2gray(img_ga)
    img_co = adjust_contrast(img_gr)
    return normalize_gr(img_co)


def transform_scale(img, ratio = None, ratio_set = [0.9, 1.1]):
    if ratio is None:
        ratio = random.sample(ratio_set, 1)[0]
    
    img_c = np.copy(img)
    rows,cols,ch = img_c.shape
    
    img1 = cv2.resize(img_c,(int(ratio*rows), int(ratio*cols)), interpolation = cv2.INTER_CUBIC)
    rows1,cols1,ch1 = img1.shape
    
    result = np.zeros_like(img_c)
    if (ratio >= 1.0):
        result = img1[(rows1 - rows)//2:(rows1 - rows)//2+rows, (cols1 - cols)//2:(cols1 - cols)//2+cols, :]
    else:
        result[(rows - rows1)//2:(rows - rows1)//2+rows1, (cols - cols1)//2:(cols - cols1)//2+cols1, :] = img1
    
    return result


def transform_rotate(img, angle = None, lower = -15,upper = 15, scale=1.0):
    if angle is None:
        angle = random.randint(lower, upper)
    
    img_c = np.copy(img)
    rows,cols,ch = img_c.shape
    center = (rows//2,cols//2) 
    
    M = cv2.getRotationMatrix2D(center,angle,scale)
    result = cv2.warpAffine(img_c,M,(rows,cols))
    
    return result


def get_trans_indices(X, y, name):
    min_number = {'tr':900, 'va': 150, 'ts':300 }
    _X_trans_idx=[]
    _set = Counter(y)
    
    for i, idx in enumerate(y):
        if(_set[idx] < min_number[name]):
            _X_trans_idx.append(i)
            _set[idx]+=1
        if(_set[idx] < min_number[name]):
            _X_trans_idx.append(i)
            _set[idx]+=1
        if(_set[idx] < min_number[name]):
            _X_trans_idx.append(i)
            _set[idx]+=1
        if(_set[idx] < min_number[name]):
            _X_trans_idx.append(i)
            _set[idx]+=1

    return Counter(_X_trans_idx)


def append_jittered_data_set2(X, y, name):

    _X_idx= get_trans_indices(X, y, name)
    
    print('>>> Scale down...')
    X_scl_dwn = [transform_scale(X[idx], ratio=0.9) for idx in _X_idx.keys() if _X_idx[idx] >= 1]
    y_scl_dwn = [y[idx] for idx in _X_idx.keys() if _X_idx[idx] >= 1]
    print('>>> Scale up...')
    X_scl_up = [transform_scale(X[idx], ratio=1.1) for idx in _X_idx.keys() if _X_idx[idx] >= 2]
    y_scl_up = [y[idx] for idx in _X_idx.keys() if _X_idx[idx] >= 2]
    print('>>> Rotate CCW...')
    X_rot_ccw = [transform_rotate(X[idx], angle=15) for idx in _X_idx.keys() if _X_idx[idx] >= 3]
    y_rot_ccw = [y[idx] for idx in _X_idx.keys() if _X_idx[idx] >= 3]
    print('>>> Rotate CW...')
    X_rot_cw = [transform_rotate(X[idx], angle=-15) for idx in _X_idx.keys() if _X_idx[idx] >= 4]
    y_rot_cw = [y[idx] for idx in _X_idx.keys() if _X_idx[idx] >= 4]
    
    print('>>> Concatenating...')
    X_jtr = np.concatenate((X, X_scl_dwn, X_scl_up, X_rot_ccw, X_rot_cw), axis=0)
    y_jtr = np.concatenate((y, y_scl_dwn, y_scl_up, y_rot_ccw, y_rot_cw), axis=0)
    print('>>> Done!\n')
    return X_jtr, y_jtr


def append_jittered_data_set(X, y):
    print('>>> Scale down...')
    X_scl_dwn = [transform_scale(img, ratio=0.9) for img in X]
    print('>>> Scale up...')
    X_scl_up = [transform_scale(img, ratio=1.1) for img in X]
    print('>>> Rotate CCW...')
    X_rot_ccw = [transform_rotate(img, angle=15) for img in X]
    print('>>> Rotate CW...')
    X_rot_cw = [transform_rotate(img, angle=-15) for img in X]
    
    print('>>> Concatenating...')
    X_jtr = np.concatenate((X, X_scl_dwn, X_scl_up, X_rot_ccw, X_rot_cw), axis=0)
    y_jtr = np.concatenate((y, y, y, y, y), axis=0)
    print('>>> Done!\n')
    return X_jtr, y_jtr


def process_data(X, y, name):
    print('>> Append the jittered data set...')
    # X_jtr, y_jtr = append_jittered_data_set(X, y)
    X_jtr, y_jtr = append_jittered_data_set2(X, y, name)
    print('>> Process the data set...')
    X_p = [image_preprocessing(img) for img in X_jtr]
    y_p = np.copy(y_jtr)
    print('>> Shuffle the data set...')
    X_p, y_p = shuffle(X_p, y_p)
    print('>> Done!\n')
    return X_p, y_p


def process_train(X_train, y_train):
    print('Training data set:')
    X_train_p, y_train_p = process_data(X_train, y_train, 'tr')
    save_processed_train_data(X_train_p, y_train_p)


def process_valid(X_valid, y_valid):
    print('Validation data set:')
    X_valid_p, y_valid_p = process_data(X_valid, y_valid, 'va')
    save_processed_valid_data(X_valid_p, y_valid_p)


def process_test(X_test, y_test):
    print('Testing data set:')
    X_test_p, y_test_p = process_data(X_test, y_test, 'ts')
    save_processed_test_data(X_test_p, y_test_p)
    

def get_test_data():
    X_test, y_test = read_test_data()
    X_test_p, y_test_p = process_test(X_test, y_test)
    return X_test_p, y_test_p


def get_variable_sizes(n_classes, stride = 1, k_size = 2, padding = 'VALID',\
                       w_in = 32, h_in = 32, d_in = 3, \
                       f0_size=8, f1_size=5, f1_d=32, l2_d=64, \
                       fc1_out = 512, fc2_out= 128):
    p = {'SAME': 1, 'VALID': 0}

    # Input layer shape
    # w_in = 32
    # h_in = 32
    # d_in = 1

    # Input filter shape
    h_f0 = w_f0 = f0_size
    d_f0 = d_in
    print('filter{0}: {1}x{2}x{3}'.format(0,h_f0,w_f0,d_f0))
    # Layer1: conv1
    w_l1 = (w_in - w_f0 + 2*p[padding]) // stride + 1
    h_l1 = (h_in - h_f0 + 2*p[padding]) // stride + 1
    d_f1 = f1_d
    d_l1 = d_f1
    print('conv{0}: {1}x{2}x{3}'.format(1,h_l1,w_l1,d_l1))
    # Maxpool 1: mp1
    w_mp1 = int(np.ceil(float((w_l1 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    h_mp1 = int(np.ceil(float((h_l1 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    d_mp1 = d_l1
    print('max_p{0}: {1}x{2}x{3}'.format(1,h_mp1,w_mp1,d_mp1))

    # Layer1 filter
    h_f1 = w_f1 = f1_size
    # d_f1 is aleady defined
    print('filter{0}: {1}x{2}x{3}'.format(1,h_f1,w_f1,d_f1))

    # Layer2: conv2
    w_l2 = (w_mp1 - w_f1 + 2*p[padding]) // stride + 1
    h_l2 = (h_mp1 - h_f1 + 2*p[padding]) // stride + 1
    d_l2 = l2_d
    print('conv{0}: {1}x{2}x{3}'.format(2,h_l2,w_l2,d_l2))
    # Maxpool 2: mp2
    w_mp2 = int(np.ceil(float((w_l2 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    h_mp2 = int(np.ceil(float((h_l2 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    d_mp2 = d_l2
    print('max_p{0}: {1}x{2}x{3}'.format(2,h_mp2,w_mp2,d_mp2))

    # Layer3: fully-connected layer (fc1)
    fc1_in = w_mp2 * h_mp2 * d_mp2
    print('fc{0}_in: {1}'.format(1,fc1_in))
    print('fc{0}_out: {1}'.format(1,fc1_out))

    # Layer4: fully-connected layer (fc2)
    fc2_in = fc1_out
    print('fc{0}_in: {1}'.format(2,fc2_in))
    print('fc{0}_out: {1}'.format(2,fc2_out))
    
    # Layer5: output layer
    out_in = fc2_out
    out_size = n_classes
    print('out_in: {0}'.format(out_in))
    print('out_size: {0}'.format(out_size))

    return w_f0, h_f0, d_f0, w_f1, h_f1, d_f1, d_l2, fc1_in, fc1_out, fc2_in, fc2_out, out_in, out_size


def get_variable_sizes2(n_classes, stride = 1, k_size = 2, padding = 'VALID',\
                       w_in = 32, h_in = 32, d_in = 3, \
                       f0_size=8, f1_size=5, f1_d=32, l2_d=64, \
                       fc1_out = 512):
    p = {'SAME': 1, 'VALID': 0}

    # Input layer shape
    # w_in = 32
    # h_in = 32
    # d_in = 1

    # Input filter shape
    h_f0 = w_f0 = f0_size
    d_f0 = d_in
    print('filter{0}: {1}x{2}x{3}'.format(0,h_f0,w_f0,d_f0))
    # Layer1: conv1
    w_l1 = (w_in - w_f0 + 2*p[padding]) // stride + 1
    h_l1 = (h_in - h_f0 + 2*p[padding]) // stride + 1
    d_f1 = f1_d
    d_l1 = d_f1
    print('conv{0}: {1}x{2}x{3}'.format(1,h_l1,w_l1,d_l1))
    # Maxpool 1: mp1
    w_mp1 = int(np.ceil(float((w_l1 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    h_mp1 = int(np.ceil(float((h_l1 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    d_mp1 = d_l1
    print('max_p{0}: {1}x{2}x{3}'.format(1,h_mp1,w_mp1,d_mp1))

    # Layer1 filter
    h_f1 = w_f1 = f1_size
    # d_f1 is aleady defined
    print('filter{0}: {1}x{2}x{3}'.format(1,h_f1,w_f1,d_f1))

    # Layer2: conv2
    w_l2 = (w_mp1 - w_f1 + 2*p[padding]) // stride + 1
    h_l2 = (h_mp1 - h_f1 + 2*p[padding]) // stride + 1
    d_l2 = l2_d
    print('conv{0}: {1}x{2}x{3}'.format(2,h_l2,w_l2,d_l2))
    # Maxpool 2: mp2
    w_mp2 = int(np.ceil(float((w_l2 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    h_mp2 = int(np.ceil(float((h_l2 + ((1 - k_size) if p[padding] is p['VALID'] else 0))) / float(k_size)))+p[padding]
    d_mp2 = d_l2
    print('max_p{0}: {1}x{2}x{3}'.format(2,h_mp2,w_mp2,d_mp2))

    # Layer3: fully-connected layer (fc1)
    fc1_in = w_mp2 * h_mp2 * d_mp2
    print('fc{0}_in: {1}'.format(1,fc1_in))
    print('fc{0}_out: {1}'.format(1,fc1_out))
    
    # Layer4: output layer
    out_in = fc1_out
    out_size = n_classes
    print('out_in: {0}'.format(out_in))
    print('out_size: {0}'.format(out_size))

    return w_f0, h_f0, d_f0, w_f1, h_f1, d_f1, d_l2, fc1_in, fc1_out, out_in, out_size

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]
        

def display_image_predictions(new_images, SignNames, my_top_k, my_softmax_logits):
    n_classes = 43
    fig, axs = plt.subplots(len(new_images), 2, figsize=(12, 40))
    fig.subplots_adjust(hspace = 0.1, wspace=2)
    axs = axs.ravel()

    for i, image in enumerate(new_images):
        guess1 = my_top_k[1][i][0]
        guess2 = my_top_k[1][i][1]
        guess3 = my_top_k[1][i][2]
        guess4 = my_top_k[1][i][3]
        guess5 = my_top_k[1][i][4]
        axs[2*i].axis('off')
        axs[2*i].imshow(image)
        axs[2*i].set_title('1st guess: {0} ({1:.2f}%)'.format(SignNames[guess1], 100*my_top_k[0][i][0]))

        axs[2*i+1].barh(np.arange(n_classes), my_softmax_logits[i])
        axs[2*i+1].set_xlabel('Softmax Prob.')
        axs[2*i+1].set_yticks(np.arange(n_classes))
        axs[2*i+1].set_yticklabels(SignNames.values())
        axs[2*i+1].set_ylim((-1,n_classes))
        axs[2*i+1].set_xlim((0,1))
        axs[2*i+1].grid('on')
        
        
