import argparse
import configparser
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

X_train = np.load(config['GSE_UNSCALED']['X_TRAIN'])
Y_train = np.load(config['GSE_UNSCALED']['Y_TRAIN'])
data_folder = config['EMBEDDING']['DATA']

parser = argparse.ArgumentParser()

#ToDo: add description
parser.add_argument('n_samples', type=int)
parser.add_argument('--offset', type=int, nargs='?', default=0)
parser.add_argument('--normalize', type=bool, nargs='?', default=True)
parser.add_argument('--valid_size', type=float, nargs='?', default=0.2)

args = parser.parse_args()

print('Selecting samples from '+str(args.offset)+' to '+str(args.offset + args.n_samples))
X = np.hstack((X_train, Y_train))
np.random.shuffle(X)
X = X[args.offset:args.offset + args.n_samples]
X = X.T

representation = '_'+str(args.n_samples)+'_'+str(args.offset)

if args.normalize:
    print('Normalizing')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    representation += '_normalized'

if args.valid_size:
    print('Assigning '+str(args.valid_size*100)+'% of genes to validation set')
    X_train, X_valid = train_test_split(X, test_size=args.valid_size)

    representation += '_'+str(args.valid_size)

    print('Saving train set with size '+str(X_train.shape))
    np.save(str(data_folder)+'/GSE92743_X_train_genes' + str(representation), X_train)

    print('Saving validation set with size ' + str(X_valid.shape))
    np.save(str(data_folder)+'/GSE92743_X_valid_genes' + str(representation), X_valid)
else:
    print('Saving train set with size ' + str(X.shape))
    np.save(str(data_folder)+'/GSE92743_X_train_genes'+str(representation), X)

