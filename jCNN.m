function CNN = jCNN(imgs,label,opts)

if isfield(opts,'nB'); num_batch = opts.nB; end
if isfield(opts,'MaxEpochs'); Maxepochs = opts.MaxEpochs; end
if isfield(opts,'LR'); LR = opts.LR; end
if isfield(opts,'kfold'); kfold = opts.kfold; end

tic;

height  = size(imgs,1);
width   = size(imgs,2); 
channel = size(imgs,3);
opts.h  = height; 
opts.w  = width; 
opts.c  = channel;

layers  = jConvolutionalStructure(opts);
options = trainingOptions('sgdm',...
    'InitialLearnRate',LR,...
    'MaxEpochs',Maxepochs,...
    'MiniBatchSize',num_batch);

fold    = cvpartition(label,'kfold',kfold);
Afold   = zeros(kfold,1); 
confmat = 0;
for i = 1:kfold
  train_idx  = fold.training(i);
  test_idx   = fold.test(i);
  xtrain     = imgs(:,:,1,train_idx);
  ytrain     = label(train_idx);
  xtest      = imgs(:,:,1,test_idx); 
  ytest      = label(test_idx);
  
  ytrain     = categorical(ytrain);
  ytest      = categorical(ytest);
  
  net        = trainNetwork(xtrain,ytrain,layers,options);
  Pred       = classify(net,xtest);
  con        = confusionmat(ytest,Pred);
  confmat    = confmat + con; 
  Afold(i,1) = sum(diag(con)) / sum(con(:));
end
Acc  = mean(Afold);
time = toc;

CNN.acc = Acc; 
CNN.con = confmat;
CNN.t   = time;

fprintf('\n Classification Accuracy (CNN): %g %% \n ',100* Acc);
end

