% List all entries in parent directory
entries = dir('../');
entries = entries([entries.isdir]); % only directories
sessionDirs = {};
for i = 1:length(entries)
    name = entries(i).name;
    if startsWith(name, '.') || startsWith(name, 'Code') || endsWith(name, '.csv')
        continue;
    end
    sessionDirs{end+1} = name; %#ok<AGROW>
end

%localDataDir = '/home/vrmaster/localdata';
localDataDir = ['G:' filesep 'Tmp'];

for s = 1:length(sessionDirs)
    currSession = sessionDirs{s};
    sessionPath = ['..' filesep currSession];

    % Show contents of session folder
    disp(['Contents of: ' sessionPath]);
    disp({dir(sessionPath).name}')  % transpose for better layout

    prompt = ['Process session "' currSession '"? (y/n): '];
    user_input = input(prompt, 's');
    if lower(user_input) ~= 'y'
        continue;
    end

    disp(['Processing session: ' currSession]);

    GoodChannels = [149 204 305 373 440 520 684 888];
    FileExt = '.dat';
    Param.FPGALength = 2588;
    Param.FrameLength = Param.FPGALength+44;
    NFrames = 1536;    
    NCounts = 2*NFrames;    
    Param.BlockSize = Param.FrameLength*NFrames;
    Param.LookUp = gray2bi_lookup;
    Param.sps = 20e3;

    % List all .dat files in session folder
    allDatFiles = dir(fullfile(sessionPath, ['*' FileExt]));
    datFiles = allDatFiles(~startsWith({allDatFiles.name}, '._') & ...
                           ~startsWith({allDatFiles.name}, '.'));
    
    disp('Found .dat files:');
    disp({datFiles.name}');
    
    if isempty(datFiles)
        warning(['No valid ' FileExt ' file found in ' currSession]);
        continue;
    end
    
    % Use first valid .dat file
    InputFileName = datFiles(1).name;
    InputFileFullPath = fullfile(sessionPath, InputFileName);
    % 
    % Clear local folder before starting
    localFiles = dir(fullfile(localDataDir, '*'));
    for k = 1:length(localFiles)
       if ~localFiles(k).isdir
            delete(fullfile(localDataDir, localFiles(k).name));
        end
    end
     
     % Copy input .dat file from NAS to local folder
    copyfile(InputFileFullPath, localDataDir);
    disp(['Copied input file to local folder: ' localDataDir]);

    % Update data directory to use the original NAS input folder
    DataDir = sessionPath;

    % Define output file name with prefix
    OutputPrefix = 'output_';
    OutputFile = [OutputPrefix InputFileName];

    bs = matlab.io.datastore.BlockedFileSet(DataDir,...
        'FileExtensions', FileExt,...
        'BlockSize',Param.BlockSize);
    disp(['Number of blocks: ' num2str(bs.NumBlocks)]); 
    t100 = 3.141877; %seconds
    disp(['Estimated run time: ' num2str(round(bs.NumBlocks*t100/100/60)) ' min']);

    NumBlocks_tmp = bs.NumBlocks;
    TotalFrames = NumBlocks_tmp*NFrames;
    TotalSamples = TotalFrames*2;
    FrameCounter = zeros(TotalFrames,1,'uint32');
    Sensors = zeros(13,TotalFrames,'uint8');
    SampleCounter = zeros(TotalSamples,1,'uint32');
    fileID = fopen([DataDir filesep OutputFile '.uint16'],'w');
    MEA1K_good = zeros(length(GoodChannels),2*TotalFrames,'uint16');

    tic
    i_t = 0;
    while hasNextBlock(bs)     
        [FrameCounter1, Sensors1, SampleCounter1, MEA1K1] = ReadBlock(nextblock(bs), Param);
        
        if isempty(FrameCounter1)
            fprintf('Stopped reading at truncated block.\n');
            break;  % Exit the reading loop
        end

        FrameCounter(i_t*NFrames+(1:NFrames)) = FrameCounter1;
        Sensors(:, i_t*NFrames+(1:NFrames)) = Sensors1;
        SampleCounter(i_t*NCounts+(1:NCounts)) = SampleCounter1;
        MEA1K_good(:,i_t*NCounts+(1:NCounts)) =  MEA1K1(1+GoodChannels,:);
        fwrite(fileID, MEA1K1,'uint16');
        if (mod(i_t+1,1000) == 0); disp(num2str(i_t+1)); end
        i_t = i_t+1;
    end
    toc

    fclose(fileID);

    FrameCounter((i_t*NFrames+1):end) = [];
    Sensors(:,(i_t*NFrames+1):end) = [];
    SampleCounter((i_t*NCounts+1):end) = [];
    MEA1K_good(:,(i_t*NCounts+1):end) = [];

    % Save parameters and supplementary data to .h5 file in session folder
    filename = [DataDir filesep OutputFile '.h5'];
    if exist(filename, 'file')>0; delete(filename); end
    h5create(filename,'/Param/FPGALength',[1 1]); h5write(filename,'/Param/FPGALength',Param.FPGALength);
    h5create(filename,'/Param/FrameLength',[1 1]); h5write(filename,'/Param/FrameLength',Param.FrameLength);
    h5create(filename,'/Param/BlockSize',[1 1]); h5write(filename,'/Param/BlockSize',Param.BlockSize);
    h5create(filename,'/Param/LookUp',size(Param.LookUp),"Datatype","uint16"); 
    h5write(filename,'/Param/LookUp',Param.LookUp);
    h5create(filename,'/Param/sps',[1 1]); h5write(filename,'/Param/sps',Param.sps);
    h5create(filename,'/FrameCounter',size(FrameCounter),"Datatype","uint32"); 
    h5write(filename,'/FrameCounter',FrameCounter);
    h5create(filename,'/Sensors',size(Sensors),"Datatype","uint8"); 
    h5write(filename,'/Sensors',Sensors);
    h5create(filename,'/SampleCounter',size(SampleCounter),"Datatype","uint32"); 
    h5write(filename,'/SampleCounter',SampleCounter);
    h5create(filename,'/MEA1K_good',size(MEA1K_good),"Datatype","uint16"); 
    h5write(filename,'/MEA1K_good', MEA1K_good);

    % Plot results (plots are saved directly to session folder)
    PlotCharts(DataDir, OutputFile, Param, FrameCounter, SampleCounter, Sensors, MEA1K1, MEA1K_good);

    % Copy output files back to NAS session folder
    % copyfile(fullfile(localDataDir, [OutputFile '.uint16']), sessionPath);
    % copyfile(fullfile(localDataDir, [OutputFile '.h5']), sessionPath);
    disp('Copied output files back to NAS session folder.');

    % Clear local folder before next session
    % localFiles = dir(fullfile(localDataDir, '*'));
    % for k = 1:length(localFiles)
    %     if ~localFiles(k).isdir
    %         delete(fullfile(localDataDir, localFiles(k).name));
    %     end
    % end
    % disp('Cleared local data folder for next session.');
end
