function [] = PlotChartsOld(DataDir, OutputFile, Param, FrameCounter, SampleCounter, Sensors, MEA1K1, MEA1K_good)
    TruncateEnd = 5; %5 last seconds to truncate do not dislay button press
    L = TruncateEnd*Param.sps/2;
    FrameCounter((end-L):end) = [];
    dFrameCounter = int32(FrameCounter(2:end)) - int32(FrameCounter(1:(end-1)));  
    NGaps = round(sum(dFrameCounter~=1)/2);
    st1 = ['Gaps in frame numbers: ' num2str(NGaps)];
    disp(st1);
    Ind = dFrameCounter~=1;
    Ind = find(Ind);
    Ind = reshape(Ind, 2,[])';
    dInd = Ind(:,2)-Ind(:,1);
    st11 = ['Too late/early: ' num2str(sum(dInd>0)) '/' num2str(sum(dInd<0))];
    disp(st11);
    st12 = ['Missed frames (median): ' num2str(median(abs(dInd))) ' Range: [' num2str(min(abs(dInd))) ' ' num2str(max(abs(dInd))) ']'];
    disp(st12);

    SampleCounter((end-2*L):end) = []; 
    dSampleCounter = int32(SampleCounter(2:end)) - int32(SampleCounter(1:(end-1)));
    Ind = (dSampleCounter==-(2^20-1)); dSampleCounter(Ind) = 1;
    NGapsSample = round(sum(dSampleCounter~=1)/2);
    st2 = ['Gaps in sample numbers: ' num2str(NGapsSample)];
    disp(st2);

    scrsz = get(0,'ScreenSize');

    % Frame counter and sample counter
    f1 = figure(1); delete(f1); 
    f1 = figure(1);
    set(f1,'Visible','off');
    set(f1,'Position', scrsz);

    subplot(4,1,1);
    X = (1:length(FrameCounter))/(Param.sps/2)/60;
    plot(X,FrameCounter,'LineWidth',1);
    xlim(X([1 end]));
    box off
    xlabel('Time (min)');
    ylabel('Frame counter');
    title([st1 '; ' st11 '; ' st12]);

    subplot(4,1,2);
    X1 = (1:length(dFrameCounter))/(Param.sps/2)/60;
    plot(X1, dFrameCounter,'LineWidth',1);
    box off
    xlim(X1([1 end]));
    ylim([-1 1]*2000);
    xlabel('Time (min)');
    ylabel('dFrame counter');

    subplot(4,1,3);
    X2 = (1:length(SampleCounter))/(Param.sps)/60;
    plot(X2,SampleCounter,'LineWidth',1);
    xlim(X2([1 end]));
    box off
    xlabel('Time (min)');
    ylabel('Sample counter');
    title(st2);

    subplot(4,1,4);
    X3 = (1:length(dSampleCounter))/(Param.sps)/60;
    plot(X3, dSampleCounter,'LineWidth',1);
    box off
    xlim(X3([1 end]));
    ylim([-1 1]*4000);
    xlabel('Time (min)');
    ylabel('dSample counter');

    set(f1,'Units','Inches');
    pos = get(f1,'Position');
    set(f1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(f1,[DataDir filesep OutputFile '_FrameCounter'],'-dpdf','-r0');

    % Battery voltage and current
    PowerInd = (Sensors(1,:)==3) | (Sensors(1,:)==4);% 4 - with illuminance
    FrameRate = Param.sps/2; %two time points in the frame
    T_power = find(PowerInd)/FrameRate;
    Power = Sensors(2:5,PowerInd);
    [V_bat, I_bat] = GetPower(Power);

    f2 = figure(2); delete(f2);
    f2 = figure(2);
    set(f2,'Visible','off');
    set(f2,'Position', scrsz);

    subplot(2,1,1);
    plot(T_power/60,V_bat);
    xlim([0, T_power(end)]/60);
    xlabel('Time (min)');
    ylabel('Battery voltage (V)');
    box off
    title(['Sampling rate ' num2str(length(T_power)/T_power(end)) ' Hz']);

    subplot(2,1,2);
    plot(T_power/60,I_bat);
    xlim([0, T_power(end)]/60);
    xlabel('Time (min)');
    ylabel('Battery current (mA)');
    box off
    title(['Mean current ' num2str(mean(I_bat)) ' mA']);

    set(f2,'Units','Inches');
    pos = get(f2,'Position');
    set(f2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(f2,[DataDir filesep OutputFile '_Battery'],'-dpdf','-r0');

    % Magnetometer, altitude, temperature
    AltCalibration = ...
    [0x01 0x81 0x6A 0xD9 0x49 0xF6 0xAB 0x00 0x99 0xF6 0x23 0x00 0x2E 0x62 0xBE 0x77 0xF3 0xF6 0x45 0x3F 0x10 0xC4 0x00 0x00 0x00 0x00 0x00];
    MagnetometerInd = (Sensors(1,:)==2);
    T_magnetometer = find(MagnetometerInd)/FrameRate;
    Magnetometer = Sensors(2:13,MagnetometerInd);
    [Magnetometer, Altitude, Temperature] = GetMagnetometerAltitudeTemper(AltCalibration, Magnetometer);

    LuxmeterInd = (Sensors(1,:)==4);
    T_luxmeter = find(LuxmeterInd)/FrameRate;
    Luxmeter = Sensors(6:9,LuxmeterInd);
    Luminosity = GetLuminosity(Luxmeter);

    f3 = figure(3); delete(f3);
    f3 = figure(3);
    set(f3,'Visible','off');
    set(f3,'Position', scrsz);

    subplot(3,1,1);
    plot(T_magnetometer/60,Altitude);
    xlim([0, T_magnetometer(end)]/60);
    xlabel('Time (min)');
    ylabel('Altitude (m)');
    box off
    title(['Sampling rate ' num2str(length(T_magnetometer)/T_magnetometer(end)) ' Hz']);

    subplot(3,1,2);
    plot(T_magnetometer/60,Temperature);
    xlim([0, T_magnetometer(end)]/60);
    xlabel('Time (min)');
    ylabel('Temperature (°C)');
    box off

    subplot(3,1,3)
    plot(T_luxmeter/60,Luminosity);
    xlim([0, T_luxmeter(end)]/60);
    xlabel('Time (min)');
    ylabel('Luminosity (lx)');
    box off

    set(f3,'Units','Inches');
    pos = get(f3,'Position');
    set(f3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(f3,[DataDir filesep OutputFile '_AltTempLum'],'-dpdf','-r0');

    % Accelerometer, gyroscope, magnetometer
    AccelerometerInd = (Sensors(1,:)==1);
    T_accelerometer = find(AccelerometerInd)/FrameRate;
    Accelerometer = Sensors(2:13,AccelerometerInd);
    [Accelerometer, Gyroscope] = GetAccelerometerGyroscope(Accelerometer);

    f4 = figure(4); delete(f4);
    f4 = figure(4);
    set(f4,'Visible','off');
    set(f4,'Position', scrsz);

    subplot(3,1,1);
    plot(T_accelerometer/60,Accelerometer');
    hl2 = legend('X','Y','Z'); 
    set(hl2,'box','off');
    xlim([0, T_accelerometer(end)]/60);
    xlabel('Time (min)');
    ylabel('Acceleration (g)');
    ylim([-1 1]*2);
    box off
    title(['Sampling rate ' num2str(length(T_accelerometer)/T_accelerometer(end)) ' Hz']);

    subplot(3,1,2);
    plot(T_accelerometer/60,Gyroscope');
    hl2 = legend('X','Y','Z'); 
    set(hl2,'box','off');
    xlim([0, T_accelerometer(end)]/60);
    xlabel('Time (min)');
    ylabel('Rotation (Rad/s)');
    ylim([-1 1]*4);
    box off
    title(['Sampling rate ' num2str(length(T_accelerometer)/T_accelerometer(end)) ' Hz']);

    subplot(3,1,3);
    plot(T_magnetometer/60,Magnetometer');
    hl2 = legend('X','Y','Z'); 
    set(hl2,'box','off');
    xlim([0, T_magnetometer(end)]/60);
    xlabel('Time (min)');
    ylabel('Magnetic field (Gauss)');
    ylim([-1 1]*4);
    box off
    title(['Sampling rate ' num2str(length(T_magnetometer)/T_magnetometer(end)) ' Hz']);

    set(f4,'Units','Inches');
    pos = get(f4,'Position');
    set(f4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(f4,[DataDir filesep OutputFile '_AccelGyroMagn'],'-dpdf','-r0');
    
    
 %4 MB of neural data to plot, approximately 1/6 sec
if ~isempty(MEA1K1)    
  for I_t = 0:7
    %I_t = 0; %from 0 to 7 to cover 8*128 = 1024 channels
    Start = I_t*128;
    V = Start+(1:128);
    MEA1K_128 = MEA1K1(V,:);
    X = (1:size(MEA1K_128,2))/Param.sps*1000; %in ms

    f5 = figure(5+I_t); delete(f5);
    f5 = figure(5+I_t); 
    set(f5,'Position', scrsz);

    plot(X,MEA1K_128');
    xlim([0, X(end)]);
    ylim([0 1024]);
    box off
    xlabel('Time (ms)');
    ylabel('ADC counts');
    title(['Channels: ' num2str(I_t*128+1) ' - ' num2str((I_t+1)*128)]);

    set(f5,'Units','Inches');
    pos = get(f5,'Position');
    set(f5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(f5,[DataDir filesep OutputFile '_MEA1K_'  num2str(I_t*128+1) '-' num2str((I_t+1)*128)],'-dpdf','-r0');
  end
end

%Compute clipped part in every good channel excluding the first and the
%last minute of record
NGoodSamplesLow = sum(MEA1K_good(:,(60*Param.sps):(end-60*Param.sps))<5);
%NGoodSamplesLow = sum(MEA1K_good(:,(5*60*Param.sps):(80*60*Param.sps))<5,2);
NGoodsamplesHigh = sum(MEA1K_good(:,(60*Param.sps):(end-60*Param.sps))>1023-5);
%NGoodsamplesHigh = sum(MEA1K_good(:,(5*60*Param.sps):(80*60*Param.sps))>1023-5,2);
NGoodSamples = NGoodSamplesLow+NGoodsamplesHigh;
NSamples = size(MEA1K_good,2)-(2*60*Param.sps-1);
%NSamples = (80-5)*60*Param.sps;
ClippedPart = double(NGoodSamples)/NSamples*100; % in percents
st = ['Clipped part (median): ' num2str(median(ClippedPart))  '% Range: [' num2str(min(ClippedPart)) ' ' num2str(max(ClippedPart)) ']%']; 
disp(st);

    X = (1:size(MEA1K_good,2))/Param.sps/60; %in min
    f13 = figure(14); delete(f13);
    f13 = figure(14); 
    set(f13,'Position', scrsz);
    plot(X,MEA1K_good');
    xlim([0, X(end)]);
    ylim([0 1024]);
    box off
    xlabel('Time (min)');
    ylabel('ADC counts');
    title(st);

    set(f13,'Units','Inches');
    pos = get(f13,'Position');
    set(f13,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(f13,[DataDir filesep OutputFile '_MEA1K_good'],'-dpdf','-r0');
end
