

function feature = mk_mfcc(x,use_logE)

if nargin<2
    use_logE = 0;
end

feature = fbank2mfcc(wav2fbank(x));
if use_logE == 1
    feature(:,13) = comp_logE(x);
end
feature(:,14:26) = comp_delta(feature,3);
feature(:,27:39) = comp_delta(feature(:,14:26),2);
