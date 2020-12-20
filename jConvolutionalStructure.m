function Model = jConvolutionalStructure(opts)

if isfield(opts,'nC'); num_conv = opts.nC; end
if isfield(opts,'FC'); fc = opts.FC; end
if isfield(opts,'nF1'); num_filt1 = opts.nF1; end
if isfield(opts,'sF1'); size_filt1 = opts.sF1; end
if isfield(opts,'nF2'); num_filt2 = opts.nF2; end
if isfield(opts,'sF2'); size_filt2 = opts.sF2; end
if isfield(opts,'nF3'); num_filt3 = opts.nF3; end
if isfield(opts,'sF3'); size_filt3 = opts.sF3; end

% Height, width, channel
h = opts.h; 
w = opts.w;
c = opts.c;

if num_conv == 1
	Model = [ imageInputLayer([h w c])
            convolution2dLayer(size_filt1,num_filt1,'Padding',1)
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'stride',2)
            fullyConnectedLayer(fc)
            softmaxLayer
            classificationLayer];
        
elseif num_conv == 2
  Model = [ imageInputLayer([h w c])
            convolution2dLayer(size_filt1,num_filt1,'Padding',1) 
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(size_filt2,num_filt2,'Padding',1) 
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'stride',2)
            fullyConnectedLayer(fc)
            softmaxLayer
            classificationLayer];
          
elseif num_conv == 3
  Model = [ imageInputLayer([h w c])
            convolution2dLayer(size_filt1,num_filt1,'Padding',1) 
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(size_filt2,num_filt2,'Padding',1) 
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(size_filt3,num_filt3,'Padding',1) 
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'Stride',2)
            fullyConnectedLayer(fc)
            softmaxLayer
            classificationLayer];
end
end

