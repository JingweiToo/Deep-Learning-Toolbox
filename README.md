# Jx-DLT : Deep Learning Toolbox

---
> "Toward Talent Scientist: Sharing and Learning Together"
>  --- [Jingwei Too](https://jingweitoo.wordpress.com/)
---


![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/eddf761e-8c8a-4866-8ee4-b8a34d541a1b/8758e6bd-bbcb-4798-9b28-19581b4a30fb/images/screenshot.PNG)


## Introduction
* This toolbox contains deep learning algorithm - Convolution neural network ( CNN ) 
* The < Main.m file > shows examples of how to use CNN programs with the benchmark data set 


## Input
* *imgs*    : feature vector ( height *x* width *x* channel *x* Instances )
* *label*   : label vector ( Instances *x* 1 )
* *opts*    : parameter settings  
  + *kfold*     : number of folds in *k*-fold cross-validation
  + *LR*        : learning rate
  + *nB*        : number of mini batch
  + *MaxEpochs* : maximum number of Epochs
  + *FC*        : number of fully connect layer ( number of classes )
  + *nC*        : number of convolutional layer ( up to 3 )
  + *nF1*       : number of filter in *1st* convolutional layer
  + *sF1*       : size of filter in *1st* convolutional layer
  + *nF2*       : number of filter in *2nd* convolutional layer
  + *sF2*       : size of filter in *2nd* convolutional layer
  + *nF3*       : number of filter in *3rd* convolutional layer
  + *sF3*       : size of filter in *3rd* convolutional layer


## Output
* *CNN* : Deep learning model ( It contains several results )  
  + *acc* : overall accuracy 
  + *con* : confusion matrix
  + *t*   : computational time (s)
  

### Example 1 : Convolutional Neural Network ( CNN ) with single convolutional layer
```code
% Benchmark dataset
[imgs,label] = digitTrain4DArrayData; 

% Parameter setting
opts.kfold     = 5;
opts.LR        = 0.01; 
opts.nB        = 100; 
opts.MaxEpochs = 20; 
opts.nC        = 1; 
opts.FC        = 10;
opts.nF1       = 16; 
opts.sF1       = [3, 3];
% Convolutional Neural Network
CNN = jCNN(imgs,label,opts);

% Accuracy
accuray = CNN.acc;
% Confusion matrix
confmat = CNN.con;
```


### Example 2 : Convolutional Neural Network ( CNN ) with three convolutional layers
```code
% Benchmark dataset
[imgs,label] = digitTrain4DArrayData; 

% Parameter setting
opts.kfold     = 5;
opts.LR        = 0.01; 
opts.nB        = 100; 
opts.MaxEpochs = 20; 
opts.nC        = 3; 
opts.FC        = 10;
opts.nF1       = 16; 
opts.sF1       = [3, 3];
opts.nF2       = 32; 
opts.sF2       = [3, 3]; 
opts.nF3       = 64; 
opts.sF3       = [3, 3];
% Convolutional Neural Network
CNN = jCNN(imgs,label,opts);

% Accuracy
accuray = CNN.acc;
% Confusion matrix
confmat = CNN.con;
```

## Requirement
* MATLAB 2017 or above
* Statistics and Machine Learning Toolbox
* Neural Network Toolbox 

## Cite As
```code
@article{too2019featureless,
  title={Featureless EMG pattern recognition based on convolutional neural network},
  author={Too, Jingwei and Abdullah, A and Saad, N Mohd and Ali, N Mohd and Zawawi, TT},
  journal={Indonesian Journal of Electrical Engineering and Computer Science},
  volume={14},
  number={3},
  pages={1291--1297},
  year={2019}
}
```
