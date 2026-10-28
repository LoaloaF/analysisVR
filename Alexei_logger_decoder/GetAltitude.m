function [ H ] = GetAltitude( P0, T0, P )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    H = ((P0./P).^(1/5.257)-1)*(T0+273.15)/0.0065;

end

