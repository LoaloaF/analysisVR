function [Voltage, Current] = GetPower(BatteryVI)
%UNTITLED2 Summary of this function goes here
%input 4xNSamples array uint8
%output voltage in volts and current in mA
%   Detailed explanation goes here
  VBUS = bitshift(uint16(BatteryVI(1,:)),8)+uint16(BatteryVI(2,:));
  VSENSE = bitshift(uint16(BatteryVI(3,:)),8)+uint16(BatteryVI(4,:));
  Voltage = double(VBUS)*9/2^16;
  Current = double(VSENSE)*500/2^16; % 100mV/0.2R = 500 mA max
end

