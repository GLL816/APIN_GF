%
% compute delta and acceleration features from static features
%
function feature = comp_dynamic_feature(feature, delta_order, acc_order)

if nargin<3
    acc_order = 2;
end
if nargin<2
    delta_order = 3;
end

D = size(feature,2);
feature(:,D+1:2*D) = comp_delta(feature(:,1:D),delta_order);
feature(:,2*D+1:3*D) = comp_delta(feature(:,D+1:2*D),acc_order);


