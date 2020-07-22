%-------------------------------------------------------------------------%
%  Deep learning algorithms source codes demo version                     %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

function Model=jConvolutionalStructure(op)
if isfield(op,'nConv'); nConv=op.nConv; end
if isfield(op,'FC'); fc=op.FC; end
if isfield(op,'nFilter1'); nFilt1=op.nFilter1; end
if isfield(op,'FilterSize1'); sFiltsSize1=op.FilterSize1; end
if isfield(op,'nFilter2'); nFilt2=op.nFilter2; end
if isfield(op,'FilterSize2'); sFiltSize2=op.FilterSize2; end
if isfield(op,'nFilter3'); nFilt3=op.nFilter3; end
if isfield(op,'FilterSize3'); sFiltSize3=op.FilterSize3; end
h=op.h; w=op.w; c=op.c;
if nConv==1
	Model=[ imageInputLayer([h w c])
          convolution2dLayer(sFiltsSize1,nFilt1,'Padding',1)
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'stride',2)
          fullyConnectedLayer(fc)
          softmaxLayer
          classificationLayer];
elseif nConv==2
  Model=[ imageInputLayer([h w c])
          convolution2dLayer(sFiltsSize1,nFilt1,'Padding',1) 
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(sFiltSize2,nFilt2,'Padding',1) 
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'stride',2)
          fullyConnectedLayer(fc)
          softmaxLayer
          classificationLayer];
elseif nConv==3
  Model=[ imageInputLayer([h w c])
          convolution2dLayer(sFiltsSize1,nFilt1,'Padding',1) 
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(sFiltSize2,nFilt2,'Padding',1) 
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(sFiltSize3,nFilt3,'Padding',1) 
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(fc)
          softmaxLayer
          classificationLayer];
end
end

