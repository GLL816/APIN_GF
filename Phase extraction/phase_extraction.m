clear all
wavpath = 'E:\������ʵ��\data\IEMOCAP\IEMOCAP\gll\newwav-VAD\session1\';
cd(wavpath);                        % dos命令cd重置当前路径，自行设置，其下包含全部待处理文�?
filelist = dir('*.wav');   % dos命令dir列出�?��的文件，用struct2cell转换为元胞数�?
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
frameLength = 0.025; %֡��
frameStep = 0.01; %֡��
framesEachSegment = 25; %ÿ�ΰ���֡��
segmentLength = frameStep*framesEachSegment+(frameLength-frameStep); %ÿ��ʱ�� 0.265s
numSeg=[];  
cd E:\������ʵ��\Matlab\��λ\�����λ
for i=1:length(filelist)
    disp(filename{i});
	wavfile = [wavpath,filename{i},ext];
    [x,fs] = audioread(wavfile);
%     x=filter([1 -0.9375],1,x); %Ԥ����
%     bank=melbankm(24,256,fs,0,0.4,'t'); %Mel�˲����Ľ���Ϊ24��FFT�任�ĳ���Ϊ256������Ƶ��Ϊ16000Hz  
%     % ��һ��Mel�˲�����ϵ��  
%     bank=full(bank); %full() convert sparse matrix to full matrix  
%     bank=bank/max(bank(:));
    %% �ֶ�
    d=segmentLength*fs;      % ÿ�β�����=segmentLength*fs
    move=frameLength*fs;
    x_start = 1;
    k=1; 
    while 1
        x_end = x_start + d-1;
        if x_end > length(x(:,1))
            break;
        end
        t = x(x_start:x_end,:);  
       % y{k}=x1;  %�ֶ����
        yy(k,:,:) = t;
        x_start = x_start + move; 
        k=k+1;
    end
    %% ����ÿ�仰���жε�phase
   % kk=numel(y);   %ÿ�����
    kk=length(yy(:,1));
    numSeg=[numSeg,kk];  %�������
    for L=1:kk
        xx=double(yy(L,:)');  
%         [S,~,~,~]=spectrogram(xx,w,ov,n,fs);
        xxx=enframe(xx,256,128);  %��֡
        [rp,ps]=relative_phase_gll(xxx,fs);
%         S=log(1+abs(S));
        z(L,:,:)=rp(1:129,:)';
    end
    DRP{i,:}=z;
    clear x_end L kk yy xx z;
end
save E:\������ʵ��\data\Paper2-phase\IEMOCAP_VAD\cell_spec\DRP_1.mat DRP