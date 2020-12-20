%---------------------------------------------------------------------%
%  Deep learning algorithm source codes demo version                  %
%---------------------------------------------------------------------%


%---Input--------------------------------------------------------------
% imgs      : feature vector (height x width x channel x instances)
% label     : label vector (instances x 1)
% kfold     : Number of cross-validation
% LR        : Learning rate
% nB        : Number of mini batch
% MaxEpochs : Maximum number of Epochs
% FC        : Number of fully connect layer (number of classes)
% nC        : Number of convolutional layer (up to 3)
% nF1       : Number of filter in first convolutional layer
% sF1       : Size of filter in first convolutional layer
% nF2       : Number of filter in second convolutional layer
% sF2       : Size of filter in second convolutional layer
% nF3       : Number of filter in third convolutional layer
% sF3       : Size of filter in third convolutional layer

%---Output-------------------------------------------------------------
% A struct that contains three results as follows:
% acc       : Overall accuracy
% con       : Confusion matrix
% t         : computational time (s)
%-----------------------------------------------------------------------


%% (1) Convolutional Neural Network with one convolutional layer
clc, clear
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


%% (2) Convolutional Neural Network with two convolutional layers
clc, clear
% Benchmark dataset
[imgs,label] = digitTrain4DArrayData; 

% Parameter setting
opts.kfold     = 5;
opts.LR        = 0.01; 
opts.nB        = 100; 
opts.MaxEpochs = 20; 
opts.nC        = 2; 
opts.FC        = 10;
opts.nF1       = 16; 
opts.sF1       = [3, 3];
opts.nF2       = 32; 
opts.sF2       = [3, 3]; 
% Convolutional Neural Network
CNN = jCNN(imgs,label,opts);

% Accuracy
accuray = CNN.acc;
% Confusion matrix
confmat = CNN.con;


%% (3) Convolutional Neural Network with three convolutional layers
clc, clear
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





