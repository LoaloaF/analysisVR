function [ SoundOut ] = ResampleBP( SoundIn, Param )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
  %tic
  N=floor(length(SoundIn)*Param.ScanRateOut/Param.ScanRateIn);
  SoundOut=int16(interpft(single(SoundIn),N,1));    %1st dimension, input 'single'  
  %toc
end

