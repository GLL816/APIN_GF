clear all
wavpath = 'E:\郭丽丽实验\data\IEMOCAP\IEMOCAP\gll\newwav-VAD\session1\';
cd(wavpath);                        % dos鍛戒护cd閲嶇疆褰撳墠璺緞锛岃嚜琛岃缃紝鍏朵笅鍖呭惈鍏ㄩ儴寰呭鐞嗘枃浠?
filelist = dir('*.wav');   % dos鍛戒护dir鍒楀嚭鎵?湁鐨勬枃浠讹紝鐢╯truct2cell杞崲涓哄厓鑳炴暟缁?
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
frameLength = 0.025; %帧长
frameStep = 0.01; %帧移
framesEachSegment = 25; %每段包含帧数
segmentLength = frameStep*framesEachSegment+(frameLength-frameStep); %每段时长 0.265s
numSeg=[];  
cd E:\郭丽丽实验\Matlab\相位\相对相位
for i=1:length(filelist)
    disp(filename{i});
	wavfile = [wavpath,filename{i},ext];
    [x,fs] = audioread(wavfile);
%     x=filter([1 -0.9375],1,x); %预加重
%     bank=melbankm(24,256,fs,0,0.4,'t'); %Mel滤波器的阶数为24，FFT变换的长度为256，采样频率为16000Hz  
%     % 归一化Mel滤波器组系数  
%     bank=full(bank); %full() convert sparse matrix to full matrix  
%     bank=bank/max(bank(:));
    %% 分段
    d=segmentLength*fs;      % 每段采样点=segmentLength*fs
    move=frameLength*fs;
    x_start = 1;
    k=1; 
    while 1
        x_end = x_start + d-1;
        if x_end > length(x(:,1))
            break;
        end
        t = x(x_start:x_end,:);  
       % y{k}=x1;  %分段组成
        yy(k,:,:) = t;
        x_start = x_start + move; 
        k=k+1;
    end
    %% 计算每句话所有段的phase
   % kk=numel(y);   %每句段数
    kk=length(yy(:,1));
    numSeg=[numSeg,kk];  %保存段数
    for L=1:kk
        xx=double(yy(L,:)');  
%         [S,~,~,~]=spectrogram(xx,w,ov,n,fs);
        xxx=enframe(xx,256,128);  %分帧
        [rp,ps]=relative_phase_gll(xxx,fs);
%         S=log(1+abs(S));
        z(L,:,:)=rp(1:129,:)';
    end
    DRP{i,:}=z;
    clear x_end L kk yy xx z;
end
save E:\郭丽丽实验\data\Paper2-phase\IEMOCAP_VAD\cell_spec\DRP_1.mat DRP