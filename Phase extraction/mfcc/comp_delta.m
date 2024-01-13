% this function compute the derivatives of the static coefficients. Its
% implementation follows that of the HMM Toolkit 3.2

function delta_coef = comp_delta(static_coef, DELTAWINDOW)

[N_vec, N_cep] = size(static_coef);

first_vec = static_coef(1,:)';
last_vec = static_coef(N_vec,:)';

static_coef = static_coef';
for i = 1:DELTAWINDOW
    static_coef = [first_vec static_coef];    % append the first feature vector DELTAWINDOW times in the front
    static_coef = [static_coef last_vec];    % append the last feature vector DELTAWINDOW times in the back
end
static_coef = static_coef';

% compute the delta coefficients
delta_coef = 0;
i = DELTAWINDOW+1;
for j = 1:DELTAWINDOW
    delta_coef = delta_coef + j*(static_coef(i+j:i+j+N_vec-1,:) - static_coef(i-j:i-j+N_vec-1,:));
end
delta_coef = delta_coef / sum((1:DELTAWINDOW).^2) * 0.5;
