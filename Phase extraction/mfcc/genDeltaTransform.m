function D = genDeltaTransform(nFr, delta_order, type)
if nargin<3
    type = 1;
end

D = zeros(nFr, nFr+delta_order*2);
for t=1 : nFr
    for j=1:delta_order
        D(t,t-j+delta_order) = -j;
        D(t,t+j+delta_order) = j;
    end
end

D2 = D(:, delta_order+1:end-delta_order);
if type==1
    D2(:,1) = sum(D(:,1:delta_order+1)')';
    D2(:,end) = sum(D(:,end-delta_order:end)')';
end

D = D2 / sum((1:delta_order).^2) * 0.5;
