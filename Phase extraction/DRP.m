function [re_phase,phase_spec] = relative_phase_gll(frames, sample_rate)
%input: frames: speech frames cut by enframe in "voicebox" 
%       sample_rate: the sample_rate for speech
%output: re_phase: relative_phase (Dynamic relative phase)
%        phase_spec: phase_spectral
%By Lili Guo

% basefreq = 1000;
NFFT = 256;
spec = fft(frames', NFFT); % spec上下对称，取一半值即可
phase_spec = angle(spec(1:NFFT/2+1,:)); %取一半spec求phas
re_phase = zeros(size(phase_spec));

% base_bin = round((NFFT / sample_rate * basefreq) - 1); %求出base_phase所在行
% base_phase = phase_spec(base_bin,:); %base_phase相位信息
% for cur_bin = 1:size(re_phase,1) % current frequency bin
%     re_phase(cur_bin,:) = phase_spec(cur_bin,:) - (double(cur_bin)/double(base_bin) * base_phase );
% end

%  re_phase(1,:) = phase_spec(1,:);
% for cur_bin = 2:size(re_phase,1) % current frequency bin
%     re_phase(cur_bin,:) = phase_spec(cur_bin,:) - (double(cur_bin)/double(cur_bin-1) * phase_spec(cur_bin-1,:) );
% end
[n,m]=size(re_phase);
for cur_bin = 1:(size(re_phase,1)-1) % current frequency bin
    re_phase(cur_bin,:) = phase_spec(cur_bin,:) - (double(cur_bin)/double(cur_bin+1) * phase_spec(cur_bin+1,:) );
end
re_phase(n,:) = phase_spec(n,:);