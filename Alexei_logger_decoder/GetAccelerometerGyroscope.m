function [Accelerometer, Gyroscope] = GetAccelerometerGyroscope(AccelGyro)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
  A = AccelGyro(6+(1:6),:);
  Accel_t = [bitshift(uint16(A(2,:)),8)+uint16(A(1,:));
          bitshift(uint16(A(4,:)),8)+uint16(A(3,:));
          bitshift(uint16(A(6,:)),8)+uint16(A(5,:))];
      
  Accel = zeros(size(Accel_t),'int16');
  for I_t = 1:3
    Accel(I_t,:) = typecast(Accel_t(I_t,:),'int16');  
  end  

  Accelerometer = single(Accel)/256/128*16; %now +/-16g, was+/-8g 

  G = AccelGyro((1:6),:);
  Gyro_t = [bitshift(uint16(G(2,:)),8)+uint16(G(1,:));
          bitshift(uint16(G(4,:)),8)+uint16(G(3,:));
          bitshift(uint16(G(6,:)),8)+uint16(G(5,:))];
      
  Gyro = zeros(size(Gyro_t),'int16');
  for I_t = 1:3
    Gyro(I_t,:) = typecast(Gyro_t(I_t,:),'int16');  
  end    
      
  Gyroscope = single(Gyro)/256/128*2000*pi/180; %2000 degrees/s -> rad/s
end

