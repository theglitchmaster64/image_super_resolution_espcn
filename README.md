
![[Pasted image 20221215121332.png]]

<center></center>
<center>Deep Learning for Image and Video Processing</center>



<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>

# Image Super Resolution using DeepLearning 


##  Abstract

Image Super Resolution is a collection of techniques used to enhance the spatial resolution of an image. This project aims to explore the current state of Image Super Resolution, in particular the techniques which make use of deep learning . Specifically, we will will train and benchmark the ESPCN (Efficient Sub-Pixel CNN), proposed by [Shi, 2016](https://arxiv.org/abs/1609.05158).
In this project, we will test the network against video frames and reconstruct short video clips to get a visual sense of how the model performs along with quantitative metrics like PSNR.


<div style="page-break-after: always;"></div>

##  Introduction

SR has many real world applications like medical imaging, survelliance, astronomical imaging, etc. SR also aids in improving other computer vision tasks and can be used as a part of a larger system. 

SR also has in applications in gaming, where new GPUs, by incorporating deep learning SR techniques as part of the rendering pipeline, can reduce the computational effort of higher resolution rendering.

<div style="page-break-after: always;"></div>

##  State of the Art

A wide variety of Deep Learning methods have been applied to this problem, from CNNs (Convolutional Neural Networks) to GANs (Generative Adversarial Networks).

The earliest methods of SR were bilinear, bicubic or nearest-neighbor interpolation. SRCNN  ([arXiv:1501.00092](https://arxiv.org/abs/1501.00092)), a deep learning approach using a CNN to represent the mapping between a LRI and it's corresponding HRI.

The model implemented here is a CNN called ESPCN or the Efficient Sub-Pixel CNN. This network is computationally more efficient than some of the other more complex architectures by extracting feature maps in the low resolution space instead of high resolution space. It also uses a filter called a sub-pixel convolution, from which it gets it's name. 

![[Pasted image 20221215063141.png]]

ESPCN was found to be `+0.15dB` better on images, and `+0.39dB` on video.
It is also magnitudes faster than previous CNN architectures like SRCNN. 

ESPCN and SRCNN are both CNNs and both use pixel-loss to train the network. Pixel loss uses the MSE (or simmilar) difference metric between the LRI and HRI. This optimizes for a metric known as PSNR or Peak Signal to Noise Ratio. PSNR is measured as a logarithmic difference metric between the HRI and LRI.

Other forms of loss functions used in different architectures are perceptual-loss, which tries to match higher level features or adversarial-loss used in GANs.


<div style="page-break-after: always;"></div>

##  Methods

An LRI can be modelled from a HRI by using a degradation function ___D___ and a noise parameter ___s___ to downsample the HRI. The HRI and LRI and then used as inputs and the expected output to the neural network, which is trained to find the inverse of the degradation we previously applied.

The degrdation function used here is gaussian blur followed by a ___decimation___. Decimate takes a signal and samples it a given rate along a given axis.

![[decimate.gif]]


```py
def degrade_image(img,scaling_factor=2):
	...
    dimg = skim.filters.gaussian(timg)
    dimg = copy.deepcopy(timg)
    dimg = scp.signal.decimate(dimg,scaling_factor,axis=0)
    dimg = scp.signal.decimate(dimg,scaling_factor,axis=1)
    ...
    return (dimg,timg,cimg,oimg)
```

The reason for using decimate over a simpler method like interpolation is because interpolation causes the downscaled image to drop tiny details, like thin straight lines and smaller objects.

The superior way is to blur the image followed by a decimation along each axis.




The ESPCN model architecture, with it's four convolutional layers and 58,212 trainable parameters.

![[model_plot.png]]
![[Pasted image 20221215072358.png]]

The activation function used in this implementation is ___relu___ instead of ___tanh___, which was used in the paper since ___relu___ achives better performance when trained for fewer epochs.
The loss function used is Mean Squared Error (MSE)


<div style="page-break-after: always;"></div>

## Dataset

I experimented with different types of images to upscale. The initial model was trained on a dataset of HD 1080p wallpapers which included landscapes and more abstract images.

![[Pasted image 20221215095649.png]] 
___HD WALLPAPERS___

The second model was trained on images extracted from a high resolution video and tested to upscale images extracted from a low resolution video.

![[Pasted image 20221215100420.png]]
__AVTR__(HRI)


![[Pasted image 20221215101034.png]]
__TNJ__ (LRI)

The problem that arose here was that the image data was unexpectedly large `>25G` and would not entire fit in memory.

To solve this I implemented batch loading and only storing filenames in memory as references instead of loading full images. 

```py
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
            psnr = float(tf.image.psnr(image, upscaled_image,max_val=255))
            print(f'mypsnr:{psnr}')
            psnr_log.append(psnr)
            skim.io.imsave(outpath+f'up{i}.png', skim.color.convert_colorspace(upscaled_image,'ycbcr','rgb'))
            skim.io.imsave(outpath+f'low{i}.png', image)
            bilinear = image.repeat(2,axis=0).repeat(2,axis=1)
            skim.io.imsave(outpath+f'blin{i}.png', image)
            psnr = float(tf.image.psnr(image, bilinear,max_val=255))
            print(f'blinpsnr:{psnr}')


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

```

Using this I was able to train the network and benchmark the PSNR.

<div style="page-break-after: always;"></div>

## Experiments

I tested the inital network trained on the HD Wallpapers against degraded AVTR images and the found the results to be visually acceptable. The results are below.

The second model was trained on AVTR (many times with different batch sizes, etc) and found a final setup with an average training loss of `0.0105608` and mean training PSNR of `58.471dB`

![[line-graph.png]]
___psnr/epoch___

![[line-graph(1).png]]
___loss/epoch___

When tested against unfamiliar data, the LRIs, the network produced results with a mean PSNR of `8.8502dB` against the bilinearly interpolated results.

I also experimented with different color spaces, specifially the `YUV` and `YCbCr` color space. 

The difference between the two is that `YUV` is a analog format and `YCbCr` is a digital format.
The model did not perform well with `YCbCr` and had color inaccuracies when any step involved the `YCbCr` color space.

Finally, I tried upscaling the LRI data again, for a total of 4X upscaling. The results show a mean PSNR of about `7.55dB`


<div style="page-break-after: always;"></div>

## Discussion

The dataset preprocessing, data pipelineing and batching created some useful insights into how to manage large datasets for DL problems. 

The results for the model's training phase are shown below:-

![[o_7090.png]]
___original__ (720p)_

![[x_7090.png]]
___degraded__ (model input, 2x downscaled, 360p)_

![[y_7090.png]]
___expected output__(720p, Y Channel of Original)_

![[b_7090.png]]
___bilinear upscaling__(720p, color?)_


The final results tested against unfamiliar data:


![[low1167.png]]
___LRI__(448 x 336)_

![[up1167.png]]
___2X Upscaled Result__(896 x 672)_

![[up101.png]]
___4X Upscaled Result __(1792 x 1344)_

As we can see, the 4X upscaled result is heavily blurred and the network doesn't perform very well when used on it's outputs.


The results for the model trained on abstract images.

![[low597.png]]
___LRI___


<div style="page-break-after: always;"></div>
![[up597.png]]
___Upscaled___


## Conclusion

This paper explored the current landscape of Image Super Resolution, summarizes image preprocessing steps, and implements a test neural network based on the ESPCN architecture. 

The project experiments with different type of image data, color spaces and double upscaling. 

We find that the PSNR metric is favourable on images the model is familiar with, but unfavourable when used on unfamiliar data.

We find that the `YUV` color space results in quicly converging loss, whereas the `YCbCr` color space results in color artefacts.

We find that double upscaling with this network does not perform well, and is almost indistingushable from bilinear interpolation.

I have gained important insights into data pipelining, training neural networks on large datasets, signal processing used in the preprocessing stage and now have a deeper understaning of sub-pixel convolutions.

Though the PSNR of the final results are not very favourable, the metric itself is not the most optimium to use.
A perception-loss  would be better suited to generate visually appealing results.
However, the entire process of data loading and preprocessing can be made faster,  and the model be more efficiently trained.


<div style="page-break-after: always;"></div>

## References

https://arxiv.org/abs/1609.05158

([arXiv:1501.00092](https://arxiv.org/abs/1501.00092)

https://stackoverflow.com/questions/49879466/downscaling-images-using-bilinear-and-bicubic-algorithms

https://dsp.stackexchange.com/questions/62177/downsample-resample-vs-antialias-fitlering-decimation

 https://keras.io/examples/vision/super_resolution_sub_pixel/#introduction
