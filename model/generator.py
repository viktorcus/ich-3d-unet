import tensorflow as tf
import keras
import os
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import random
import skimage
import imblearn
import keras.backend as K


transform = ['horizontal_flip', 'vertical_flip', 'blur', 'gaussian_noise', 'rotation', 'affine', 'dropout', 'scale_axes', 'translate', 'scale_axes', 'rotation']


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, items, dim, batch_size, img_dir, mask_dir, partition, complete=False,
                validation=False, n_channels=1, shuffle=True, window=False,
                 undersampling=True, sampling_ratio=0.5, split=False, augment=True, ones_only=False):
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.window = window
        self.undersampling=undersampling
        self.sampling_ratio=sampling_ratio
        self.partition=partition
        self.split = split
        self.augment = augment
        self.ones_only = ones_only
        self.complete = complete
        
        
            
        if(validation):
            self.filenames = items[:self.partition]
        else:
            self.filenames = items[self.partition:]

        self.positives = np.zeros(len(self.filenames))
        if undersampling or self.ones_only:
            self.positives = self.has_positives()

        if self.complete:
            self.batch_size = len(self.filenames)
        else:
            self.batch_size = batch_size

    def __len__(self):
        if self.split:
            return len(self.filenames) * 2
        else:
            return len(self.filenames)
        
    def on_epoch_end(self):
        if self.shuffle:
            z = list(zip(self.filenames, self.positives))
            random.shuffle(z)
            self.filenames, self.positives = zip(*z)

    def __data_generation(self):
        if not self.split:
            dim2 = 58
            batch = self.batch_size
        elif self.complete:
            dim2 = self.dim[2]
            batch = self.batch_size * 2
        else:
            dim2 = self.dim[2]
            batch = self.batch_size
        
        images = np.zeros(shape=(batch, self.dim[0], self.dim[1], dim2))
        masks = np.zeros(shape=(batch, self.dim[0], self.dim[1], dim2))

        x = self.filenames
        y = np.zeros(len(x))
        if self.undersampling:
            rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={0: 20, 1: 6}, replacement=True)
            x, y = rus.fit_resample(np.array(x).reshape(-1, 1), self.positives)
            x = np.reshape(x, (len(x),))
        elif self.ones_only:
            rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={0: 0, 1: 20}, replacement=True)
            x, y = rus.fit_resample(np.array(x).reshape(-1, 1), self.positives)
            x = np.reshape(x, (len(x),))

        z = list(zip(x, y))
        random.shuffle(z)
        x, y = zip(*z)

        i = 0
        while i < self.batch_size:
            img_path = os.path.join(self.img_dir, x[i])
            img = np.load(img_path)

            mask_path = os.path.join(self.mask_dir, x[i])
            mask = np.load(mask_path)

            if self.augment:
                img, mask = self.augment_image(img, mask)

            if self.split:
                if self.complete: 
                    images[i*2], masks[i*2], images[i*2+1], masks[i*2+1] =  self.split_middle(img, mask, False, True)
                elif self.ones_only:
                    images[i], masks[i] = self.split_middle(img, mask, True, False)
                elif self.undersampling:
                    images[i], masks[i] = self.split_middle(img, mask, y[i], False)
                else:
                    images[i], masks[i] = self.split_middle(img, mask, False, False)
            else:
                images[i] = img
                masks[i] = mask

            if (self.ones_only and y[i]) or not self.ones_only:
                i = i + 1;


        for mask in masks:
          mask[mask > 0] = 1

        images = images[:, :, :, :, np.newaxis]
        masks = masks[:, :, :, :, np.newaxis]

        return images, masks

    def split_middle(self, img, mask, take_pos=False, take_both=False):
        i1 = np.zeros((251, 251, 32))
        i2 = np.zeros((251, 251, 32))
        m1 = np.zeros((251, 251, 32))
        m2 = np.zeros((251, 251, 32))

        half = random.randrange(0, 2)
        if not take_both:
          slices = []
          if take_pos:
            for j in range(58):
              if mask[:,:,j].any():
                slices.append(j)
          slices = np.array(slices)
          if take_pos and [s for s in slices if s < 16 or s >= 42] and not [s for s in slices if s >= 13 or s < 45]:
            half = 0
          elif take_pos and (not [s for s in slices if s < 16 or s >= 42]) and [s for s in slices if s >= 13 or s < 45]:
            half = 1
          
        if take_both:
            i1[:,:,0:16] = img[:,:,0:16]
            i1[:,:,16:32] = img[:,:,42:58]
            m1[:,:,0:16] = mask[:,:,0:16]
            m1[:,:,16:32] = mask[:,:,42:58]
            i2 = img[:,:,13:45]
            m2 = mask[:,:,13:45]
            return i1, m1, i2, m2
        elif half == 0:
            i1[:,:,0:16] = img[:,:,0:16]
            i1[:,:,16:32] = img[:,:,42:58]
            m1[:,:,0:16] = mask[:,:,0:16]
            m1[:,:,16:32] = mask[:,:,42:58]
            return i1, m1
        else:   # half == 1:
            i1 = img[:,:,13:45]
            m1 = mask[:,:,13:45]
            return i1, m1


    def augment_image(self, image, mask):
        aug = random.sample(transform, random.randint(2,4))

        def select_aug(aug, image, mask):
          if aug == 'horizontal_flip':
            augmenter = iaa.flip.HorizontalFlip(1)
            return augmenter(image=image), augmenter(image=mask)
          elif aug == 'vertical_flip':
            augmenter = iaa.flip.VerticalFlip(1)
            return augmenter(image=image), augmenter(image=mask)
          elif aug == 'blur':
            augmenter = iaa.blur.GaussianBlur(sigma=random.uniform(0.0, 1.3))
            return augmenter(image=image), mask
          elif aug == 'gaussian_noise':
            im = skimage.util.random_noise(image, mode='gaussian', seed=None)
            return im, mask
          elif aug == 'rotation':
            angle = random.uniform(-180, 180)
            return skimage.transform.rotate(image, angle=angle), skimage.transform.rotate(mask, angle=angle)
          elif aug == 'affine':
            scale = random.uniform(0.7, 1.3)
            augmenter_img = iaa.size.KeepSizeByResize(iaa.geometric.Affine(scale=scale, order=3))
            augmenter_mask = iaa.size.KeepSizeByResize(iaa.geometric.Affine(scale=scale, order=0))
            return augmenter_img(image=image), augmenter_mask(image=mask)
          elif aug == 'dropout':
            image = image.astype('float32')
            augmenter = iaa.arithmetic.Dropout(p=random.uniform(0.0, 0.25))
            return augmenter(image=image).astype('float64'), mask
          elif aug == 'scale_axes':
            scale_x = random.uniform(0.7, 1.3)
            scale_y = random.uniform(0.7, 1.3)
            augmenter_img = iaa.size.KeepSizeByResize(iaa.geometric.Affine(scale={"x": scale_x, "y": scale_y}, order=3))
            augmenter_mask = iaa.size.KeepSizeByResize(iaa.geometric.Affine(scale={"x": scale_x, "y": scale_y}, order=0))
            return augmenter_img(image=image), augmenter_mask(image=mask)
          elif aug == 'translate':
            translate_x = random.randint(-20, 20)
            translate_y = random.randint(-20, 20)
            augmenter_img = iaa.size.KeepSizeByResize(iaa.geometric.Affine(translate_px={"x": translate_x, "y": translate_y}, order=3))
            augmenter_mask = iaa.size.KeepSizeByResize(iaa.geometric.Affine(translate_px={"x": translate_x, "y": translate_y}, order=0))
            return augmenter_img(image=image), augmenter_mask(image=mask)

        for a in aug:
          image, mask = select_aug(a, image, mask)
        
        return image, mask


    def has_positives(self):
        b = np.zeros(len(self.filenames))
        for i in range(len(self.filenames)):
            path = os.path.join(self.mask_dir, self.filenames[i])
            mask = np.load(path)
            b[i] = mask.any()
        return b
    
    def __getitem__(self, index):
        z = list(zip(self.filenames, self.positives))
        np.random.shuffle(z)
        self.filenames, self.positives = zip(*z)
        self.filenames = list(self.filenames)
        i, m = self.__data_generation()


        return i, m