% comp_log_Mel receives a time domain speech signal and produces its
% log Mel filterbank coefficients. It is the Matlab counterpart of the
% feature extraction program provided with AURORA2 database. 
% Author: Xiao Xiong
% Created: 18 Jul, 2005
% Last Modified: 28 Jul, 2005
% Inputs: 
%       x   1-D time domain signal
% Outputs: 
%       hist_log_mel    the log Mel filterbank coefficients
%       hist_mel        the Mel filterbank coefficients
%       hist_abs        the spectral magnitude
%       logE            the log energy item needed in computing MFCC

% the fine version of wav2abs have half of the normal frame_shift, i.e.,
% its frame_shift is 5ms instead of 10ms.
function [Abs_x, fft_x] = wav2abs(x,Fs,frame_shift, frame_size, FFT_length)

if nargin<4
    frame_size = Fs * 0.025;
else
    frame_size = Fs * frame_size;
end
if nargin<3
    frame_shift = Fs * 0.01;
else
    frame_shift = Fs * frame_shift;
end
if nargin < 2
    Fs = 8000;
end

if nargin<5
    FFT_length = pow2(ceil(log2(frame_size)));
end

% FFT
fft_x = sfft(x,frame_size,frame_shift,Fs,FFT_length);
% fft_x=sfft(x);
% get the magnitude
Abs_x = abs(fft_x(2:FFT_length/2+1,:))';