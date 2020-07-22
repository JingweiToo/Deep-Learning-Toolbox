%-------------------------------------------------------------------------%
%  Deep learning algorithms source codes demo version                     %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

function CNN=jCNN(imgs,label,op)
if isfield(op,'nBatch'); nBatch=op.nBatch; end
if isfield(op,'MaxEpochs'); Maxepochs=op.MaxEpochs; end
if isfield(op,'LR'); LR=op.LR; end
if isfield(op,'kfold'); kfold=op.kfold; end
height=size(imgs,1); width=size(imgs,2); channel=size(imgs,3);
op.h=height; op.w=width; op.c=channel;
layers=jConvolutionalStructure(op);
options=trainingOptions('sgdm','InitialLearnRate',LR,...
    'MaxEpochs',Maxepochs,'MiniBatchSize',nBatch);
fold=cvpartition(label,'kfold',kfold);
Afold=zeros(kfold,1); confmat=0;
for i=1:kfold
  trainIdx=fold.training(i); testIdx=fold.test(i);
  xtrain=imgs(:,:,1,trainIdx); ytrain=label(trainIdx);
  xtest=imgs(:,:,1,testIdx); ytest=label(testIdx);
  ytrain=categorical(ytrain); ytest=categorical(ytest);
  net=trainNetwork(xtrain,ytrain,layers,options);
  Pred=classify(net,xtest);
  con=confusionmat(ytest,Pred);
  confmat=confmat+con; 
  Afold(i,1)=100*sum(diag(con))/sum(con(:));
end
Acc=mean(Afold); 
CNN.fold=Afold; CNN.acc=Acc; CNN.con=confmat; 
fprintf('\n Classification Accuracy (CNN): %g %% \n ',Acc);
end

