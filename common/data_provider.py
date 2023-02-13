import os
import pickle
import numpy as np

class DataProvider:
    __HOME = os.path.expanduser('~')
    __DATAROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data'))
    __DATA2DROOT = os.path.join(__DATAROOT, 'data2d')
    DATA2DROOT_TRAIN = os.path.join(__DATA2DROOT, 'data2d_train')
    DATA2DROOT_TEST = os.path.join(__DATA2DROOT, 'data2d_test')
    _MNISTROOT = os.path.join(__DATAROOT, 'mnist')
    MNIST_TRAIN = os.path.join(_MNISTROOT, 'train')
    MNIST_TEST = os.path.join(_MNISTROOT, 'test')

    def __init__(self, dataset, verbose=False):
        if verbose:
            print('Loading database')
        in_filename = '%s.p' % dataset
        with open(in_filename, 'rb') as in_fp:
            self.__dataset_dict = pickle.load(in_fp, encoding='latin1')
        if verbose:
            print('Database loaded')

    def get_class_arr(self, class_idx):
        return self.__dataset_dict[str(class_idx)]

    def get_dataset_arr(self):
        class_arr_list = [self.__dataset_dict[key] for key in sorted(self.__dataset_dict.keys())]
        return np.vstack(tuple(class_arr_list))

    def get_dataset_and_labels(self, class_idx=None):
        if class_idx is None:
            labels = []
            for class_id, dataset in sorted(self.__dataset_dict.items()):
                size = dataset.shape[0]
                labels.extend([class_id] * size)
            return self.get_dataset_arr(), np.hstack((labels))
        else:
            data = []
            labels = []
            for class_id in class_idx:
                d = self.get_class_arr(class_id)
                data.append(d)
                labels.extend([str(class_id)] * d.shape[0])
            data = np.vstack((data))
            labels = np.array(labels)
            return data, labels

    @staticmethod
    def write_data(path, dataset_dict):
        path = '%s.p' % path
        with open (path, 'wb') as out_fp:
            pickle.dump(dataset_dict, out_fp)

