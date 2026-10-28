function [Magnetometer, Altitude, Temperature] = GetMagnetometerAltitudeTemper(C, MagnetometerPresureTemp)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

M = MagnetometerPresureTemp(1:6,:);
Magn_t = [bitshift(uint16(M(2,:)),8)+uint16(M(1,:));
        bitshift(uint16(M(4,:)),8)+uint16(M(3,:));
        bitshift(uint16(M(6,:)),8)+uint16(M(5,:))];

Magn = zeros(size(Magn_t),'int16');
  for I_t = 1:3
    Magn(I_t,:) = typecast(Magn_t(I_t,:),'int16');  
  end 
    
Magnetometer = single(Magn)/256/128*49.152; %49.152 Gauss    
            

%This is for BMP388:
Calibration.dig_T1 = uint16(C(2))+bitshift(uint16(C(3)),8); %unsigned
Calibration.dig_T2 = uint16(C(4))+bitshift(uint16(C(5)),8); %unsigned
Calibration.dig_T3 = typecast(uint8(C(6)),'int8');
Calibration.dig_P1 = typecast(uint16(C(7))+bitshift(uint16(C(8)),8),'int16');
Calibration.dig_P2 = typecast(uint16(C(9))+bitshift(uint16(C(10)),8),'int16');
Calibration.dig_P3 = typecast(uint8(C(11)),'int8');
Calibration.dig_P4 = typecast(uint8(C(12)),'int8');
Calibration.dig_P5 = uint16(C(13))+bitshift(uint16(C(14)),8);
Calibration.dig_P6 = uint16(C(15))+bitshift(uint16(C(16)),8);
Calibration.dig_P7 = typecast(uint8(C(17)),'int8');
Calibration.dig_P8 = typecast(uint8(C(18)),'int8');
Calibration.dig_P9 = typecast(uint16(C(19))+bitshift(uint16(C(20)),8),'int16');
Calibration.dig_P10 = typecast(uint8(C(21)),'int8');
Calibration.dig_P11 = typecast(uint8(C(22)),'int8');

P = MagnetometerPresureTemp(6+(1:3),:)';
Pressure_t = int32(bitshift(uint32(P(:,3)),16)+bitshift(uint32(P(:,2)),8)+bitshift(uint32(P(:,1)),0));
P = MagnetometerPresureTemp(9+(1:3),:)';
Temp_t = int32(bitshift(uint32(P(:,3)),16)+bitshift(uint32(P(:,2)),8)+bitshift(uint32(P(:,1)),0));

[ Temperature, P ] = GetCalibTemperaturePressureBMP388( Temp_t, Pressure_t, Calibration );  %Celsius, Pascals

%Values at zero level to compute relative elevation, taken from Zurich
P0 = 96733.5;   %Pa
T0 = 28.2;      %degrees C

[ Altitude ] = GetAltitude( P0, T0, P );  %Celsius, Pascals; altitude meters

end

