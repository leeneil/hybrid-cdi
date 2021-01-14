# hybrid-cdi
The low-resolution guided (LRG) method for X-ray coherent diffraction imaging 

Paper: https://arxiv.org/abs/2004.03017

## Usage

First, save the diffraction **UN-FFT-SHIFTED** in a numpy file. Then run:

```
python reconstruct.py input_image.npy
```

To reconsruct it with our LRG method, run

```
python reconstruct.py input_image.npy --algo lrg
```

