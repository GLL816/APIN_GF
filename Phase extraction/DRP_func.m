function [re_phase,phase_spec] = DRP_func(frames, sample_rate)
%input: frames: speech frames cut by enframe in "voicebox" 
%       sample_rate: the sample_rate for speech
%output: re_phase: relative_phase (Dynamic relative phase)
%        phase_spec: phase_spectral


% basefreq = 1000;
NFFT = 256;
spec = fft(frames', NFFT); % spec
phase_spec = angle(spec(1:NFFT/2+1,:)); 
re_phase = zeros(size(phase_spec));

% base_bin = round((NFFT / sample_rate * basefreq) - 1); 
% base_phase = phase_spec(base_bin,:); 
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
