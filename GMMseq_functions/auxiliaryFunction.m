function [Q, gradientQ] = auxiliaryFunction(x, y, K, T, L, time_repT, initmodel)
%[Q, gradientQ] = auxiliaryFunction(x, y, K, T, L, time_repT, initmodel)
% Part of function Q that depends on gamma, tau and beta
% Inputs follows the same nomenclature as GMMSEQ_train.m
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
alpha = beta ./ ( 1.0 + exp( - gamma .* (time_repT - tau) ) );
alpha(:,1) = 1.0;% property eq 4
monpi = alpha ./ sum(alpha, 2);
% fprintf('%d pts 0\n',sum( sum(monpi,2)==0 ));

% Q value
Q = y .* log( monpi + double(monpi==0)); 
assert(~any(isnan(Q(:)))); 
assert(~any(isinf(Q(:))));
Q = sum(Q,'all');

% with penalisation
if ~isempty(initmodel.penalisation)
    switch initmodel.penalisation.type
        case 'l1'
            error('not implemented yet');
        case 'l2'
            Q = Q - initmodel.penalisation.lambda * sum( power( tau - initmodel.penalisation.tauprior, 2 ), 'omitnan');
    end
end

% to be used in fminunc return negative of the Q (to be maximised)
Q = -Q;

% Gradient of Q
if nargout > 1 
    
    if initmodel.useMiniBatches
        r = randperm(L);
        r = r(1:min(L,initmodel.sizeMiniBatches));
        y = y(r,:);
        time_repT = time_repT(r,:);
        L = length(r);
        monpi = monpi(r,:);
        alpha = alpha(r,:);
    end
    
    gradientQ = compute_gradient(y, time_repT, K, T, monpi, alpha, beta, gamma, tau, initmodel);
    gradientQ = -gradientQ;
end



