function [Luminosity] = GetLuminosity(Regs)
%UNTITLED Summary of this function goes here
%input parameter is 4xNsamples matrix uint8
% output is luminosity in Lux
%   Detailed explanation goes here
  CPL = 2.73*3*16/60;
  C0DATA = single(uint16(Regs(2,:))*256+uint16(Regs(1,:)));
  C1DATA = single(uint16(Regs(4,:))*256+uint16(Regs(3,:)));
  Lux1 = (C0DATA-1.87*C1DATA)/CPL;
  Lux2 = (0.63*C0DATA-C1DATA)/CPL;
  Luminosity = max(Lux1, 0);
  Luminosity = max(Luminosity, Lux2);
end

