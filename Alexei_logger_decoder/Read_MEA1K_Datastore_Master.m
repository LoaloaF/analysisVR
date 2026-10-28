% List all entries in parent directory
%Mydatapath = ['g:' filesep 'Remote' filesep 'OldFormat']; % line 35 must match
%Mydatapath = ['g:' filesep 'Remote' filesep 'NewFormat'];
%Mydatapath = ['g:' filesep 'Remote' filesep 'BothFormats']; %for debugging at local comp.
% Mydatapath = ['x:' filesep 'BMI' filesep 'VirtualReality' filesep ...
%     'SpatialSequenceLearning' filesep 'RUN_rYL010' filesep ...
%     'rYL010_P0000']; %to access NAS when NAS is mapped to X: drive

Mydatapath = '..'; % simon

entries = dir(Mydatapath); % just for testing
%entries = dir('../');
entries = entries([entries.isdir]); % only directories
sessionDirs = {};
for i = 1:length(entries)
    name = entries(i).name;
    if startsWith(name, '.') || startsWith(name, 'Code') || endsWith(name, '.csv') || startsWith(name, "_")
        continue;
    end
    sessionDirs{end+1} = name; % #ok<AGROW>
end

localDataDir = '/run/media/vrmaster/data2/tmpproc'
% localDataDir = '/home/vrmaster/localdata';
% localDataDir = ['F:' filesep 'Tmp'];

FileExt = '.dat';


for s = 1:length(sessionDirs)
    currSession = sessionDirs{s};
    sessionPath = [Mydatapath filesep currSession];

    % Show contents of session folder
    disp(['Contents of: ' sessionPath]);
    disp({dir(sessionPath).name}')  % transpose for better layout

    prompt = ['Process session "' currSession '"? (y/n): '];
     user_input = input(prompt, 's');
     if lower(user_input) ~= 'y'
         continue;
     end

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


    disp(['Processing session: ' currSession]);
    Param.Format = GetFormat(sessionPath); %his should be implemented
    % 1 - 20250519
    % 2 - 20250525

    GoodChannels = [149 204 305 373 440 520 684 888];
    Param.FPGALength = 2588;
    Param.FrameLength = Param.FPGALength+44;
    if (Param.Format==2); Param.FrameLength = Param.FrameLength+28; end
    NFrames = 1536;    
    NCounts = 2*NFrames;    
    Param.BlockSize = Param.FrameLength*NFrames;
    Param.LookUp = gray2bi_lookup;
    Param.sps = 20e3;



    % % % Use first valid .dat file - bad idea, all data files should be copied
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
    for i_t = 1:length(datFiles) 
        copyfile([Mydatapath filesep currSession filesep datFiles(i_t).name], localDataDir);
        disp(['Copied input files to local folder: ' localDataDir]);
    end

    [~, OutputBase, ~] = fileparts(InputFileName);
    OutputFile = OutputBase;
    
    % Create single HDF5 output file named after the .dat file
    filename = [sessionPath filesep OutputFile '_ephys_logger.raw.h5'];
    if exist(filename, 'file')>0; delete(filename); end
    h5create(filename,'/MEA1K_raw', [Inf 1024], 'Datatype','uint16','ChunkSize',[512 1024],'Deflate',5);

    bs = matlab.io.datastore.BlockedFileSet(sessionPath,...
        'FileExtensions', FileExt,...
        'BlockSize',Param.BlockSize);
    disp(['Number of blocks: ' num2str(bs.NumBlocks)]); 
    t100 = 12; %seconds
    disp(['Estimated run time: ' num2str(round(bs.NumBlocks*t100/100/60)) ' min']);

    NumBlocks_tmp = bs.NumBlocks;
    TotalFrames = NumBlocks_tmp*NFrames;
    TotalSamples = TotalFrames*2;
    FrameCounter = zeros(TotalFrames,1,'uint32');
    Sensors = zeros(13,TotalFrames,'uint8');
    SampleCounter = zeros(TotalSamples,1,'uint32');
    MEA1K_good = zeros(length(GoodChannels),2*TotalFrames,'uint16');
    if (Param.Format==2); Sound = zeros(TotalFrames*20,1,'int16'); end
    Sync.X = []; Sync.Y = []; 

    tic
    i_t = 0;
    while hasNextBlock(bs)     
        [FrameCounter1, Sensors1, SampleCounter1, MEA1K1, Sync1, Sound1] = ReadBlock(nextblock(bs), Param);

        if (i_t==0) && (~isempty(Sync1.StartY))
           Sync.X = 1;
           Sync.Y = Sync1.StartY;       
        end
        if ~isempty(Sync1.X)
           Sync.X = [Sync.X  i_t*NFrames*100+Sync1.X];
           Sync.Y = [Sync.Y  Sync1.Y];  
        end

        FrameCounter(i_t*NFrames+(1:NFrames)) = FrameCounter1;
        Sensors(:, i_t*NFrames+(1:NFrames)) = Sensors1;
        if (Param.Format==2); Sound(i_t*NFrames*20+(1:(NFrames*20))) = Sound1; end
        SampleCounter(i_t*NCounts+(1:NCounts)) = SampleCounter1;
        MEA1K_good(:,i_t*NCounts+(1:NCounts)) =  MEA1K1(1+GoodChannels,:);

        % Write block of MEA1K1 to HDF5 dataset
        % h5write(filename, '/MEA1K_raw', MEA1K1, [1 1 + 2*i_t*NFrames], size(MEA1K1));
        h5write(filename, '/MEA1K_raw', MEA1K1.', [1 + 2*i_t*NFrames, 1], size(MEA1K1.'));
        if (mod(i_t+1,1000) == 0); disp(num2str(i_t+1)); end
        i_t = i_t+1;
    end
    toc

    FrameCounter((i_t*NFrames+1):end) = [];
    Sensors(:,(i_t*NFrames+1):end) = [];
    SampleCounter((i_t*NCounts+1):end) = [];
    MEA1K_good(:,(i_t*NCounts+1):end) = [];

    if ~isempty(Sync.X)      
        Sync.X = [Sync.X  i_t*NFrames*100];
        Sync.Y = [Sync.Y  Sync.Y(end)];   
    end

    % Save metadata
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
    if ~isempty(Sync.X)    
        ds = '/SyncX'; h5create(filename,ds,size(Sync.X),"Datatype","double"); 
                      h5write(filename,ds,Sync.X);
        ds = '/SyncY'; h5create(filename,ds,size(Sync.Y),"Datatype","uint8"); 
                      h5write(filename,ds,uint8(Sync.Y));    
    end
    if (Param.Format==2)
        ds = '/Sound'; h5create(filename,ds,size(Sound),"Datatype","int16"); 
                      h5write(filename,ds,Sound);   
    end

    % Plot results into logger_plots subfolder
    plotDir = fullfile(sessionPath, 'logger_plots');
    if ~exist(plotDir, 'dir'); mkdir(plotDir); end
    disp(['📊 Writing plots to: ' plotDir]);
    if (Param.Format==1)
        PlotChartsOld(plotDir, OutputFile, Param, FrameCounter, SampleCounter, Sensors, MEA1K1, MEA1K_good);
    end    
    if (Param.Format==2)
        PlotCharts(plotDir, OutputFile, Param, FrameCounter, SampleCounter, Sensors, MEA1K1, Sync, Sound, MEA1K_good);
    end


    % % Copy output files back to NAS session folder
    %  copyfile(fullfile(localDataDir, [OutputFile '.uint16']), sessionPath); %path correction is needed here
    %  copyfile(fullfile(localDataDir, [OutputFile '.h5']), sessionPath);
    % disp('Copied output files back to NAS session folder.');

    % Clear local folder before next session
    % localFiles = dir(fullfile(localDataDir, '*'));
    % for k = 1:length(localFiles)
    %     if ~localFiles(k).isdir
    %         delete(fullfile(localDataDir, localFiles(k).name));
    %     end
    % end
    % disp('Cleared local data folder for next session.');
end
