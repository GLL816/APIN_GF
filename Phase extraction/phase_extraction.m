clear all
wavpath = 'E:\GLL\data\IEMOCAP\IEMOCAP\gll\newwav-VAD\session1\';
cd(wavpath);                        
filelist = dir('*.wav');   
filelist = struct2cell(filelist);
filelist = filelist(1,:)';
filename = cell(length(filelist),1);
 
w=256;     
ov=w/2;   
% w=400;     
% ov=240; 
n=256; 

for i=1:length(filelist)
    a = filelist(i);
    [pathstr,name,ext] = fileparts(a{1});
    filename{i,1} = name;
end
% spec = zeros(length(filelist),k,1024);
frameLength = 0.025; 
frameStep = 0.01; 
framesEachSegment = 25; 
segmentLength = frameStep*framesEachSegment+(frameLength-frameStep); 
numSeg=[];  
cd E:\GLL\Matlab\phase\relative phase
for i=1:length(filelist)
    disp(filename{i});
	wavfile = [wavpath,filename{i},ext];
    [x,fs] = audioread(wavfile);

    %% segment
    d=segmentLength*fs;     
    move=frameLength*fs;
    x_start = 1;
    k=1; 
    while 1
        x_end = x_start + d-1;
        if x_end > length(x(:,1))
            break;
        end
        t = x(x_start:x_end,:);  
        yy(k,:,:) = t;
        x_start = x_start + move; 
        k=k+1;
    end


    kk=length(yy(:,1));
    numSeg=[numSeg,kk];  
    for L=1:kk
        xx=double(yy(L,:)');  
        xxx=enframe(xx,256,128);  
        [rp,ps]=DRP_fuc(xxx,fs);
        z(L,:,:)=rp(1:129,:)';
    end
    DRP{i,:}=z;
    clear x_end L kk yy xx z;
end
save E:\GLL\data\Paper2-phase\IEMOCAP_VAD\cell_spec\DRP.mat DRP
