function [FrameCounter, Sensors, SampleCounter, MEA1K, Sync, Sound] = ReadBlock(blkInfo, Param)
% Outputs: FrameCounter, Sensors, SampleCounter, MEA1K

% Open file and seek to block offset
fr = matlab.io.datastore.DsFileReader(blkInfo.Filename);
seek(fr, blkInfo.Offset);
D = read(fr, Param.BlockSize);

% === Check for incomplete read ===
if mod(numel(D), Param.FrameLength) ~= 0
    warning('ReadBlock: Truncated data block. Stopping read.');
    FrameCounter = [];
    Sensors = [];
    SampleCounter = [];
    MEA1K = [];
    return;
end


% Reshape into [FrameLength x NumFrames]
D = reshape(D, Param.FrameLength, []);

% === Extract FrameCounter ===
Position = Param.FrameLength - 19;  % without IR, "-14" previously
if (Param.Format==2); Position = Position-28; end %with IR and sound;
FrameCounter8 = D(Position + (1:4), :);
FrameCounter = uint32(FrameCounter8(1, :)) + ...
               bitshift(uint32(FrameCounter8(2, :)), 8) + ...
               bitshift(uint32(FrameCounter8(3, :)), 16) + ...
               bitshift(uint32(FrameCounter8(4, :)), 24);

if (Param.Format==2)
    %Sound
    Sound_t =  D(Position+4+(1:40),:);
    Sound_t = reshape(Sound_t, 2, []);
    Sound_t = bitshift(uint16(Sound_t(2,:)),8) + uint16(Sound_t(1,:));
    Sound = typecast(Sound_t, 'int16');            
else
    Sound = [];
end

% === Extract Sensors ===
Sensors = D(Param.FPGALength + 12 + (1:13), :);  % without IR, no +12
if (Param.Format==2)    %in fact sync can exist here, but omit to save time in current data sets
    Sync_in = D(Param.FPGALength-6+(1:13),:);
    [Sync.X, Sync.Y, Sync.StartY] = GetSync(Sync_in);
else
    Sync.X = []; Sync.Y = []; Sync.StartY = [];
end

% === Prepare MEA1K data ===
MaxCount = 1032 * 2 * 5 / 4;
B = D(2 + (1:MaxCount), :);
clear D;

% Reshape to [MaxCount/2 x numFrames]
B = reshape(B, MaxCount / 2, []);

% Convert from 10-bit packed format (reshape into [5 x N])
B = reshape(B, 5, []);
D = zeros(4, size(B, 2), 'uint16');
D(1, :) = uint16(B(1, :)) + bitshift(uint16(bitand(B(5, :), 0x03)), 8);
D(2, :) = uint16(B(2, :)) + bitshift(uint16(bitand(B(5, :), bitshift(0x03, 2))), 6);
D(3, :) = uint16(B(3, :)) + bitshift(uint16(bitand(B(5, :), bitshift(0x03, 4))), 4);
D(4, :) = uint16(B(4, :)) + bitshift(uint16(bitand(B(5, :), bitshift(0x03, 6))), 2);
clear B;

% Reshape to [1032 x numFrames]
D = reshape(D, 1032, []);

% === Extract SampleCounter ===
SampleCounter2 = D(1024 + (1:2), :);
SampleCounter = uint32(SampleCounter2(1, :)) + bitshift(uint32(SampleCounter2(2, :)), 10);

% === Extract MEA1K data ===
D(1025:end, :) = [];  % Keep only 1024 rows (data)
D = reshape(D, 1, []);
D = Param.LookUp(D + 1);  % Apply lookup
MEA1K = reshape(D, 1024, []);
end
