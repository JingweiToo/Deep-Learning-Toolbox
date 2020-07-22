%-------------------------------------------------------------------------%
%  Deep learning algorithm source codes demo version                     %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%


%---Input------------------------------------------------------------------
% imgs:        features (height x width x channel x instances)
% label:       labelling (labels x 1)
% kfold:       Number of cross-validation
% LR:          Learning rate
% nBatch:      Number of mini batch
% MaxEpochs:   Maximum number of Epochs
% FC:          Number of fully connect layer (number of classes)
% nConv:       Number of convolutional layer (up to 3)
% nFilter1:    Number of filter in first convolutional layer
% nFilteSize1: Size of filter in first convolutional layer
% nFilter2:    Number of filter in second convolutional layer
% nFilteSize2: Size of filter in second convolutional layer
% nFilter3:    Number of filter in third convolutional layer
% nFilteSize3: Size of filter in third convolutional layer
%---Output-----------------------------------------------------------------
% A struct that contains three results as follows:
% fold: Accuracy for each fold
% acc:  Average accuracy over k-folds
% con:  Confusion matrix
%--------------------------------------------------------------------------

%% (1) Convolutional Neural Network with one convolutional layer
clc, clear
% Benchmark dataset
[imgs,label]=digitTrain4DArrayData; 
% Parameter setting
op.kfold=5; op.LR=0.01; op.nBatch=100; op.MaxEpochs=20; 
op.nConv=1; op.FC=10; op.nFilter1=16; op.FilterSize1=[3,3];
% Convolutional Neural Network
CNN1=jCNN(imgs,label,op);


%% (2) Convolutional Neural Network with two convolutional layers
clc, clear
% Benchmark dataset
[imgs,label]=digitTrain4DArrayData; 
% Parameter setting
op.kfold=5; op.LR=0.01; op.nBatch=100; op.MaxEpochs=20;
op.nConv=2; op.FC=10; op.nFilter1=16; op.FilterSize1=[3,3]; op.nFilter2=32; 
op.FilterSize2=[3,3]; 
% Convolutional Neural Network
CNN2=jCNN(imgs,label,op);


%% (3) Convolutional Neural Network with three convolutional layers
clc, clear
% Benchmark dataset
[imgs,label]=digitTrain4DArrayData; 
% Parameter setting
op.kfold=5; op.LR=0.01; op.nBatch=100; op.MaxEpochs=20;
op.nConv=3; op.FC=10; op.nFilter1=16; op.FilterSize1=[3,3]; op.nFilter2=32; 
op.FilterSize2=[3,3]; op.nFilter3=64; op.FilterSize3=[3,3];
% Convolutional Neural Network
CNN3=jCNN(imgs,label,op);







