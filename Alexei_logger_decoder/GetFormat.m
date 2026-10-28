function [Format] = GetFormat(sessionPath)
% Outputs: FrameCounter, Sensors, SampleCounter, MEA1K

%compute frame length suitable for both cases
 Param.FPGALength = 2588;
 Param.FrameLength1 = Param.FPGALength+44; %2632/4 = 658
 Param.FrameLength2 = Param.FrameLength1+28; %2660/4 = 665
 FSize = round(Param.FrameLength1*Param.FrameLength2/4); % 1750280

%find dat file
 FileExt = '.dat';
 allDatFiles = dir(fullfile(sessionPath, ['*' FileExt]));
 datFiles = allDatFiles(~startsWith({allDatFiles.name}, '._') & ...
                           ~startsWith({allDatFiles.name}, '.'));
if ~isempty(datFiles)
    FName = [sessionPath filesep datFiles(1).name];
    fileID = fopen(FName);
    A = fread(fileID,[1, FSize],'*uint8');
    fclose(fileID);
end 

% Reshape into [FrameLength x NumFrames]
A1 = reshape(A, Param.FrameLength1, []);
A2 = reshape(A, Param.FrameLength2, []);

% === Extract FrameCounter ===
Position1 = Param.FrameLength1 - 19;  % without IR, "-14" previously
Position2 = Position1-28; %with IR and sound;
FrameCounter18 = A1(Position1 + (1:4), :);
FrameCounter28 = A2(Position2 + (1:4), :);
FrameCounter1 = uint32(FrameCounter18(1, :)) + ...
               bitshift(uint32(FrameCounter18(2, :)), 8) + ...
               bitshift(uint32(FrameCounter18(3, :)), 16) + ...
               bitshift(uint32(FrameCounter18(4, :)), 24);
FrameCounter2 = uint32(FrameCounter28(1, :)) + ...
               bitshift(uint32(FrameCounter28(2, :)), 8) + ...
               bitshift(uint32(FrameCounter28(3, :)), 16) + ...
               bitshift(uint32(FrameCounter28(4, :)), 24);
dFrameCounter1 = FrameCounter1(2:end)-FrameCounter1(1:(end-1));
dFrameCounter2 = FrameCounter2(2:end)-FrameCounter2(1:(end-1));
s1 = sum(dFrameCounter1~=1);
s2 = sum(dFrameCounter2~=1);
if s1>s2 %more mistakes in the first
    Format = 2;
else
    Format = 1;
end

end