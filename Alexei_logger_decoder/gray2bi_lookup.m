function [LookUp] = gray2bi_lookup()
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
X = zeros(1024,1,'uint16');
for i=1:1024
   V = de2bi(uint16(i-1),10,'left-msb');
   V1 = gray2bi(V);
   X(i) = bi2de(V1,'left-msb');
end
LookUp = X;
end