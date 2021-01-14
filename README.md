# Hybrid real- and reciprocal-space full-field imaging with coherent illumination
The low-resolution guided (LRG) method for X-ray coherent diffraction imaging 

Authors: Po-Nan Li, Soichi Wakatsuki, Piero A. Pianetta, Yijin Liu
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

## Contact

For bugs, issues or suggestions, file an issue on GitHub or contact Po-Nan Li at liponan@ponan.li

