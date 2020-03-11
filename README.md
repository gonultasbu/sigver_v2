
# Offline Handwritten Signature Verification Using Machine Learning

This repository is a spinoff from https://github.com/luizgh/sigver with some architectural tweaks and domain adaptations.

# Usage

## Data preprocessing

The workflow is as follows: images > apply data preprocessing > .npz files > apply CNN feature extractions > .csv features > apply writer-dependent classifiers > scores. 

The functions in this package expect training data to be provided in a single .npz file, with the following components:

* ```x```: Signature images (numpy array of size N x 1 x H x W)
* ```y```: The user that produced the signature (numpy array of size N )
* ```yforg```: Whether the signature is a forgery (1) or genuine (0) (numpy array of size N )

We provide functions to process some commonly used datasets in the script ```sigver.datasets.process_dataset```. 
As an example, the following code pre-process the MCYT dataset with the procedure from [1] (remove background, center in canvas and resize to 170x242)

```bash
python -m sigver.preprocessing.process_dataset --dataset mcyt \
 --path MCYT-ORIGINAL/MCYToffline75original --save-path mcyt_170_242.npz
```

During training a random crop of size 150x220 is taken for each iteration. During test we use the center 150x220 crop.

## Training a CNN for Writer-Independent feature learning

This repository implements the two loss functions defined in [1]: SigNet (learning from genuine signatures only)
and SigNet-F (incorporating knowledge of forgeries). In the training script, the flag
```--users``` is use to define the users that are used for feature learning. In [1],
GPDS users 300-881 were used (```--users 300 881```). 


Training SigNet:

```
python -m sigver.featurelearning.train --model signet --dataset-path  <data.npz> --users [first last]\ 
--model signet --epochs 60 --logdir signet  
```

Training SigNet-F with lambda=0.95 with L1 loss type:

```
python -m sigver.featurelearning.train --model signet --dataset-path  <data.npz> --users [first last]\ 
--model signet --epochs 60 --forg --lamb 0.95 --logdir signet_f_lamb0.95 --loss-type L1  
```

For checking all command-line options, use ```python -m sigver.featurelearning.train --help```. 
In particular, the option ```--visdom-port``` allows real-time monitoring using [visdom](https://github.com/facebookresearch/visdom) (start the visdom
server with ```python -m visdom.server -port <port>```).   

# Pre-trained models

Pre-trained models can be found here: 
* SigNet ([link](https://storage.googleapis.com/luizgh-datasets/pytorch_models/signet.pth))
* SigNet-F lambda 0.95 ([link](https://storage.googleapis.com/luizgh-datasets/pytorch_models/signet_f_lambda_0.95.pth))

Or simply download via command line:

These models contains the weights for the feature extraction layers.

**Important**: These models were trained with pixels ranging from [0, 1]. Besides the pre-processing steps described above, you need to divide each  pixel by 255 to be in the range. This can be done as follows: ```x = x.float().div(255)```. Note that Pytorch does this conversion automatically if you use ```torchvision.transforms.totensor```, which is used during training.

Usage:

```python

import torch
from sigver.featurelearning.models import SigNet

# Load the model
state_dict, classification_layer, forg_layer = torch.load('models/signet.pth')
base_model = SigNet().eval()
base_model.load_state_dict(state_dict)

# Extract features
with torch.no_grad(): # We don't need gradients. Inform torch so it doesn't compute them
    features = base_model(input)

```

# Citation

If you use the code, please consider citing the following papers:

[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))


# License

The source code is released under the BSD 3-clause license. Note that the trained models used the GPDS dataset for training, which is restricted for non-commercial use.  

# TODO

Generate fake signatures using GANs.  
Robustness against Gaussian noise (Generality).  




