#!/usr/bin/python3

import numpy as np
import skimage as skim
import scipy as scp
import random
import sys
import os
import copy
from random import shuffle

import time

import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

def degrade_image(img,scaling_factor=2):
    img = img[0:720,0:960]
    oimg = copy.deepcopy(img)
    img = img.astype('float32')
    img = img / 255.0
    timg = skim.color.convert_colorspace(img, 'RGB', 'yuv')
    ru = timg[:,:,2].repeat(2,axis=0).repeat(2,axis=1)
    rv = timg[:,:,1].repeat(2,axis=0).repeat(2,axis=1)
    timg = timg[:,:,0]
    dimg = skim.filters.gaussian(timg)
    dimg = copy.deepcopy(timg)
    dimg = scp.signal.decimate(dimg,scaling_factor,axis=0)
    dimg = scp.signal.decimate(dimg,scaling_factor,axis=1)
    cimg = np.dstack((dimg,ru,rv))
    cimg = skim.color.convert_colorspace(cimg, 'yuv', 'rgb')
    return (dimg,timg,cimg,oimg)

def upscale_img(model, img):
    timg = img.astype('float32')/255.0
    timg = skim.color.convert_colorspace(img, 'RGB','yuv')
    y, cb, cr = np.dsplit(timg,3)

    # use Y channel for upscaling
    y = np.squeeze(y)
    model_in = np.expand_dims(y,axis=0) # change shape from (x,y) to (x,y,1)
    model_out = model.predict(model_in)
    new_y = np.squeeze(model_out)

    print(new_y.shape)

    #bilinear upscaling for other channels
    cr = np.squeeze(cr)
    cb = np.squeeze(cb)
    cr = cr.repeat(2,axis=0).repeat(2,axis=1)
    cb = cb.repeat(2,axis=0).repeat(2,axis=1)

    #merge channels
    outimg = np.dstack((new_y,cb,cr))
    return outimg


def create_model(upscale_factor=2,channels_in=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels_in))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels_in * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)
    return keras.Model(inputs, outputs)


# batch sizes for 40040 = 8, 10, 40,
#1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 20, 22, 26, 28, 35, 40, 44, 52, 55, 56, 65, 70, 77, 88, 91, 104, 110, 130, 140, 143, 154, 182, 220, 260, 280, 286, 308, 364, 385, 440, 455, 520, 572, 616, 715, 728, 770, 910, 1001, 1144, 1430, 1540, 1820, 2002, 2860, 3080, 3640, 4004, 5005, 5720, 8008, 10010, 20020, 40040

class DataLoader:
    def __init__(self, dir='./img_seq'):
        dir = dir.strip('/')
        self.file_list = os.listdir(dir)
        self.file_list.sort()
        self.file_list = list(map(lambda x: f'{dir}/{x}',self.file_list))
        self.n = len(self.file_list)

    def get_batch(self, batch_size, n_batch):
        total_batches = len(self.file_list)//batch_size
        batch = {'X':[], 'Y':[]}
        for i in range(batch_size * n_batch ,batch_size * (n_batch+1) ):
            print(i)
            x,y,c,o = degrade_image(skim.io.imread(self.file_list[i]))
            batch['X'].append(x)
            batch['Y'].append(y)
            shuffle(batch['X'])
            shuffle(batch['Y'])
        return ( np.array(batch['X']), np.array(batch['Y']) )

    def get_random_validation(self):
        rand = random.randint(len(self.file_list)-101,len(self.file_list)-1)
        _, _, c, o = degrade_image(skim.io.imread(self.file_list[rand]))
        return (c,o)

    def batch_upscale(self, model, start,end , outpath):
        psnr_log = []
        for i in range(start,end):
            print(f'batch_upscale:{i}')
            image = skim.io.imread(self.file_list[i])
            upscaled_image = upscale_img(model, image)
            upscaled_image *= 255.0
            skim.io.imsave(outpath+f'up{i}.png', skim.color.convert_colorspace(upscaled_image,'yuv','rgb'))
            skim.io.imsave(outpath+f'low{i}.png', image)
            bilinear = image.repeat(2,axis=0).repeat(2,axis=1)
            skim.io.imsave(outpath+f'blin{i}.png', image)
            psnr = float(tf.image.psnr(upscaled_image, bilinear,max_val=255))
            psnr_log.append(psnr)
            print(f'psnr:{psnr}')
        return psnr_log


    def demo(self, start, end, outpath):
        for i in range(start,end):
            print(f'demo:{i}')
            x,y,c,o = degrade_image(skim.io.imread(self.file_list[i]))
            skim.io.imsave(outpath+f'x_{i}.png',x)
            skim.io.imsave(outpath+f'y_{i}.png',y)
            skim.io.imsave(outpath+f'c_{i}.png',c)
            skim.io.imsave(outpath+f'o_{i}.png',o)
            #do billinear upscaling
            b = c.repeat(2,axis=0).repeat(2,axis=1)
            skim.io.imsave(outpath+f'b_{i}.png',b)


    def __len__(self):
        return self.n

    def __repr__(self):
        return f'{self.file_list}'


def train_model(model, dataset, batch_size, n_batches, epochs, savepath='./models/model_',tag=None):
    metadata = {'psnr':[]}
    if batch_size * n_batches > len(dataset.file_list):
        return f'error could not train model with {batch_size} x {n_batches}'
    savefile = savepath+str(int(time.time()))+f'_enbt_{epochs},{n_batches},{batch_size},{epochs*n_batches,batch_size}'
    if tag != None:
        savefile +='_'+ str(tag)
    total_PSNR = 0
    for i in range(1,epochs+1):
        print(f'epoch: {i}')
        for j in range(n_batches):
            X, Y = dataset.get_batch(batch_size, j)
            print(f'fitting batch {j}; epoch {i}')
            model.fit(X,Y)
            #psnr
            lri, hri = dataset.get_random_validation()
            upscaled_lri = upscale_img(model,lri)
            epoch_psnr = tf.image.psnr(upscaled_lri, hri,max_val=255)
            total_PSNR += epoch_psnr
            mean_psnr = total_PSNR / (i*(j+1))
            print(f'epoch_psnr={epoch_psnr}; mean_psnr = {mean_psnr}')
            metadata['psnr'].append(epoch_psnr)
    print(f'saving model to {savefile}')
    model.save(savefile)
    return metadata


if __name__=='__main__':
    print('supm8')

    print('creating model...')
    model = create_model()
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss=loss_fn)
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print('done')

    print('load data...')
    #dataset = DataLoader('./img_seq')
    #dataset2 = DataLoader('./img_seq2')

    print('begin train/load model')
    #charts = train_model(model, dataset, 10, 1000, 1)
    model = keras.models.load_model('models/model0')

    #dataset.demo(7000,8000,outpath='./outputs/avtr/')
    #chart = dataset2.batch_upscale(model, 700,1700, outpath='./outputs/tnj/')
    dataset3 = DataLoader('./img_seq2/')
    dataset3.batch_upscale(model, 250,750, outpath='./outputs/tnj/hdwalls_out/')
