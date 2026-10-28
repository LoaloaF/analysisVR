function [ T, P ] = GetCalibTemperaturePressureBMP388( T0, P0, C )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
  var1 = double(T0)-double(C.dig_T1)*256;
  var2 = var1*double(C.dig_T2)/2^30;
  t_fine = var2+var1.^2*double(C.dig_T3)/2^48;    %no round, 2^48
  T = t_fine; %in Celsius
  
  var1 = double(C.dig_P6)/2^6*t_fine;
  var2 = double(C.dig_P7)/2^8*t_fine.^2;
  var3 = double(C.dig_P8)/2^15*t_fine.^3; 
  var1out = double(C.dig_P5)*2^3+var1+var2+var3;
  
  var1 = (double(C.dig_P2)-2^14)/2^29*t_fine;
  var2 = double(C.dig_P3)/2^32*t_fine.^2;
  var3 = double(C.dig_P4)/2^37*t_fine.^3; 
  var2out = double(P0).*((double(C.dig_P1)-2^14)/2^20+var1+var2+var3);
  
  var1 = double(P0).^2;
  var2 = double(C.dig_P9)/2^48+double(C.dig_P10)/2^48*t_fine;
  var3 = var1.*var2;
  var4 = var3+double(P0).^3*double(C.dig_P11)/2^65;
  P = var1out+var2out+var4; %in Pascals
end

