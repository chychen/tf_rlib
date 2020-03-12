import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tf_rlib import datasets
from absl import flags, logging
import nibabel as nib
from os import listdir, system
from skimage.transform import rescale

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

class NTUH_HeartArota(datasets.Dataset):
    '''
    This dataset is private in NTUH.
    '''
    def __init__(self, path='/mount/src/home/ntun/Whole_heart/NTUHv2/clara/data/', max_crop_size = [144, 144, 144]):
        self.path = path
        self.max_crop_size = max_crop_size
        super(NTUH_HeartArota, self).__init__()
        
    def get_data(self):
        return self._get_dsets()
    
    def _get_dsets(self):
        train_path = self.path+'train/'
        valid_path = self.path+'val/'
        imgs_train, segs_train, _ = self._load_data(train_path)
        imgs_valid, segs_valid, _ = self._load_data(valid_path)
        imgs_train, segs_train = self._preprocess(imgs_train, segs_train)
        imgs_valid, segs_valid = self._preprocess(imgs_valid, segs_valid)
        gen_train = self._get_generator(imgs_train, segs_train)
        gen_valid = self._get_generator(imgs_valid, segs_valid, val=True)
        self.gen_train = gen_train
        self.gen_valid = gen_valid
        train_dataset = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32),
            (tf.TensorShape([None, None, None, None, 1]), tf.TensorShape([None, None, None, None, 3])))
        valid_dataset = tf.data.Dataset.from_generator(
            gen_valid, (tf.float32, tf.float32),
            (tf.TensorShape([None, None, None, None, 1]), tf.TensorShape([None, None, None, None, 3])))
        train_dataset = train_dataset.cache().shuffle(20).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.cache().prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_dataset, valid_dataset]
        
    def _load_data(self, path):
        img_fnames, seg_fnames = [], []
        for fname in listdir(path):
            if '.nii.gz' in fname:
                if 'segs' not in fname:
                    img_fnames.append(fname)
                else:
                    seg_fnames.append(fname)
        imgs, segs = [], []
        for fname in img_fnames:
            img = nib.load(path+fname).get_fdata()
            seg = np.squeeze(nib.load(path+[f for f in seg_fnames if fname[:9] in f][0]).get_fdata())
            imgs.append(img), segs.append(seg)
        return imgs, segs, img_fnames
    
    def _preprocess(self, imgs, segs):
        imgs_ = [self._pre_transform(img[..., None]) for img in imgs]
        segs_ = [self._pre_transform(seg, isLabel=True) for seg in segs]
        return imgs_, segs_
    
    def _pre_transform(self, x, isLabel=False):
        # scale
        '''
        order:
          0: Nearest-neighbor
          1: Bi-linear 
        '''
        if isLabel:
            order = 0
        else:
            order = 1
        x = rescale(x, .5, order=order, multichannel=True)

        return x
    
    def _get_generator(self, x, y, val=False):
        def gen():
            idx = np.arange(len(x))
            np.random.shuffle(idx)
            for i in range(len(x) // FLAGS.bs):
                batch_x, batch_y = self._sample_data(x, y, idx[i*FLAGS.bs:(i+1)*FLAGS.bs], val=val)
                yield batch_x, batch_y

        return gen
    
    def _sample_data(self, x, y, idx, val=False):
        xs, ys = [], []
        for i in range(FLAGS.bs):
            if val:
                _x, _y = self._val_transform([x[idx[i]], y[idx[i]]])
            else:
                _x, _y = self._aug_transform([x[idx[i]], y[idx[i]]], min_crop_size=[48, 48, 48])
            xs += [_x]
            ys += [_y]
        return np.array(xs), np.array(ys)
    
    def _aug_transform(self, data, min_crop_size):
        x, y = data
        # normalize
        '''
        Normalize input to zero mean and unit standard deviation, based on non-zero elements only
        '''
        x = (x-x.min())/(x.max()-x.min())
        for c in range(x.shape[-1]):
            mu, std = x[..., c][x[..., c]!=0].mean(), x[..., c][x[..., c]!=0].std()
            x[..., c] = (x[..., c]-mu)/std
        # random crop
        '''
        x: [h, w, d, c]
        '''
        size = [np.random.randint(min_crop_size[i], x.shape[i] if x.shape[i]<self.max_crop_size[i] else self.max_crop_size[i]) for i in range(len(x.shape)-1)] + [x.shape[-1]+y.shape[-1]]
        data = np.concatenate([x, y], axis=-1)
        data = tf.image.random_crop(data, size).numpy()
        x = data[..., 0][..., None]
        y = data[..., 1:]
        # random flip
        flip_axis = np.random.randint(0, len(x.shape))
        if flip_axis==len(x.shape)-1:
            pass
        else:
            x = np.flip(x, axis=flip_axis)
            y = np.flip(y, axis=flip_axis)
        # random scale and shift intensity
        scale = np.random.uniform(0, 0.1)
        shift = np.random.uniform(-0.1, 0.1)
        x = x * (1 + scale) + shift * np.std(x)
        # pad to be divisible by 16
        x = self._pad_div_by_n(x, n=16)
        y = self._pad_div_by_n(y, n=16)
        return x, y
    
    def _val_transform(self, data):
        x, y = data
        # normalize
        '''
        Normalize input to zero mean and unit standard deviation, based on non-zero elements only
        '''
        x = (x-x.min())/(x.max()-x.min())
        for c in range(x.shape[-1]):
            mu, std = x[..., c][x[..., c]!=0].mean(), x[..., c][x[..., c]!=0].std()
            x[..., c] = (x[..., c]-mu)/std
        # crop
        hd, hw, hh = [s//2 for s in self.max_crop_size]
        size = x.shape[:3]
        x = x[size[0]//2-hd if size[0]//2-hd>0 else 0:size[0]//2+hd,
              size[1]//2-hd if size[1]//2-hd>0 else 0:size[1]//2+hd,
              size[2]//2-hd if size[2]//2-hd>0 else 0:size[2]//2+hd]
        y = y[size[0]//2-hd if size[0]//2-hd>0 else 0:size[0]//2+hd,
              size[1]//2-hd if size[1]//2-hd>0 else 0:size[1]//2+hd,
              size[2]//2-hd if size[2]//2-hd>0 else 0:size[2]//2+hd]
        # pad to be divisible by 16
        x = self._pad_div_by_n(x, n=16)
        y = self._pad_div_by_n(y, n=16)
        return x, y
    
    def _pad_div_by_n(self, x, n=16):
        '''
        x: [h, w, d, c]
        '''
        ori_size = x.shape[:3]
        new_size = [s+16-s%16 for s in ori_size]
        paddings = np.array([[(n-o)//2, (n-o)-(n-o)//2] for o, n in zip(ori_size, new_size)]+[[0, 0]])
        new_x = tf.pad(x, paddings, 'CONSTANT', constant_values=0).numpy()
        return new_x