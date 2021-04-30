function [alpha, monpi, pi_phi, y] = Estep(x, K, data, t_repT, T, mu, sigma, useExternalFunctions)
%[alpha, monpi, pi_phi, y] = Estep(x, K, data, t_repT, T, mu, sigma, useExternalFunctions)
% Perform the Estep given a model GMMSEQ.
% useExternalFunctions = true to use external functions instead of Matlab 
% internal ones.
% [x = [b(:) g(:) xi(:)] are parameters for k=2...K (instrumental
% variables)
% T = max(t), L = length(t), time_repT is the time vector replicated over
% clusters (dim NxK).
% Inputs follows the same nomenclature as GMMSEQ_train.m
%
%   REFERENCE
%   
%   This implementation has been made and shared in the context of a 
%   project between FEMTO-ST (Besançon, France) and UTC (Compiègne, France)
%   which yielded the following paper:
%
%   [1] Emmanuel Ramasso, Thierry Denoeux, Gael Chevallier, Clustering 
%   acoustic emission data stream with sequentially appearing clusters 
%   using mixture models, Mechanical Systems and Signal Processing, 2021.
%
%
% Emmanuel Ramasso and Thierry Denoeux
% emmanuel.ramasso@femto-st.fr
% April 2021



[~,~,~,beta,gamma,tau] = getParamFromX(x,K,T);

% comput alpha
alpha = beta ./ ( 1.0 + exp( - gamma .* (t_repT - tau) ) );
alpha(:,1) = 1.0;% property eq 4
monpi = alpha ./ sum(alpha, 2);

% compute y
phi = zeros(size(monpi), 'double');
for k=1:K
     if not(useExternalFunctions)
        phi(:,k) = mvnpdf(data, mu(k,:), squeeze(sigma(k,:,:)));
    else
        phi(:,k) = mixgauss_prob(data', mu(k,:)', squeeze(sigma(k,:,:)));
    end
end
pi_phi = monpi .* phi;

% points belong to nothing?!
s = sum(pi_phi,2);
f = find(s == 0); 
pi_phi(f,:) = ones(length(f),K) / K;
s(f) = 1.0;

y = pi_phi ./ s;% normalise

