function [gradientQ,g,b,xi] = compute_gradient(y, t_repT, K, T, monpi, alpha, beta, gamma, tau, initmodel)
%[gradientQ,g,b,xi] = compute_gradient(y, t_repT, K, L, T, monpi, alpha, beta, gamma, tau, initmodel)
% Compute the gradients of GMMSEQ model
% Inputs follows the same nomenclature as GMMSEQ_train.m
% Is called in auxiliary_function.m
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




dQ_dg = zeros(1,K,'double');
dQ_db = zeros(1,K,'double');
dQ_dxi = zeros(1,K,'double');

% Following matrices are Lx(K-1) => indexing is from k-1:K, k=2...K
% dalpha_dbeta, for k=2...K, for t=1...L
alpha_tk_over_betak = alpha(:,2:end) ./ beta(:,2:end); % eq 13

% dalpha_dtau, for k=2...K, for t=1...L
unmoinsalphaoverbeta = (1.0 - alpha_tk_over_betak);
dalpha_tk_dtau_k = -gamma(:,2:end) .* alpha(:,2:end) .* unmoinsalphaoverbeta; % eq 15

% dalpha_dgamma, for k=2...K, for t=1...L
dalpha_tk_dgamma_k = alpha(:,2:end) .* (t_repT(:,2:end) - tau(:,2:end)) .* unmoinsalphaoverbeta; % eq 14


%%%%%%%%%%%%%%%%%%%%%%
% compute derivatives
% dpi wrt dalpha and dQ

% This is a matrix T x K
% monpi can be 0 in that cas dQ_dpi = nan, will be managed below
% monpi = monpi + double(monpi==0);
dQ_dpi = y ./ monpi;% eq 11

x = setXfromParam(struct('beta',beta,'gamma',gamma,'tau',tau,'nb_clusters',K,'T',T));
b = [nan x(1:K-1)];
g = [nan x(K:2*K-2)];
xi = [nan x(2*K-1:3*K-3)];

% dpi wrt dalpha 
alpha_all_norm_k = sum(alpha,2); % alpha(:,1)=1
for_k_neq_l = -alpha ./ (alpha_all_norm_k .^ 2);
q = 1.0 ./ alpha_all_norm_k;
%for_k_eq_l = 1 ./ alpha_all_norm_k - alpha ./ (alpha_all_norm_k .^ 2);
% for_k_eq_l = 1 ./ alpha_all_norm_k + for_k_neq_l;

% dpi_dalpha = cell(K,1);
% for l=1:K
%     dpi_dalpha{l} = for_k_neq_l;% eq 12 - we are going to overwrite the values for which k=l below
%     dpi_dalpha{l}(:,l) = for_k_eq_l(:,l);
% end

%%%%%%%%%%%%%%%%%%%
% Lambda = @(u) 1 ./ (1 + exp(-u) );
% Ftauk = @(u) T*Lambda(u); % eq 14a

% select valid dQ_dpi (w/o nan)
% f = isnan(dQ_dpi);
% if any(f,'all')
%     fprintf('Gradients: %d pts nan\n',sum(f,'all')); 
% end
% dQ_dpi(f) = 0.0;

for k=2:K
    
    %dQ_dpi_times_dpi_dalpha = dQ_dpi .* dpi_dalpha{k};% common factor
    dQ_dpi_times_dpi_dalpha = for_k_neq_l;
    dQ_dpi_times_dpi_dalpha(:,k) = dQ_dpi_times_dpi_dalpha(:,k) + q;
    dQ_dpi_times_dpi_dalpha = dQ_dpi .* dQ_dpi_times_dpi_dalpha;% common factor
    
    dQ_dbetak =  dQ_dpi_times_dpi_dalpha .* alpha_tk_over_betak(:,k-1);% eq 10a
    dQ_dbetak = sum( dQ_dbetak, 2, 'omitnan');
    
    dQ_dgammak = dQ_dpi_times_dpi_dalpha .* dalpha_tk_dgamma_k(:,k-1);% eq 10b
    dQ_dgammak = sum( dQ_dgammak, 2, 'omitnan');
    
    dQ_dtauk = dQ_dpi_times_dpi_dalpha .* dalpha_tk_dtau_k(:,k-1);% eq 10c
    dQ_dtauk = sum( dQ_dtauk, 2, 'omitnan');
    
    % with penalisation
    if ~isempty(initmodel.penalisation)
        switch initmodel.penalisation.type
            case 'l1'
                error('not implemented yet');
            case 'l2'
                if ~isnan(initmodel.penalisation.tauprior(k))
                    dQ_dtauk = dQ_dtauk - 2.0*initmodel.penalisation.lambda .* ( tau(k) - initmodel.penalisation.tauprior(k) );
                end
            otherwise
                error('l2-type penalty only is implemented')
        end
    end
    
    dQ_dg(k) = 2.0*g(k)*sum( dQ_dgammak, 1, 'omitnan');% eq 9a
    dQ_db(k) = 2.0*b(k)*sum( dQ_dbetak, 1, 'omitnan');% eq 9b
    dQ_dxi(k) = sum( dQ_dtauk, 1, 'omitnan') * tau(k)  * (1.0 - tau(k)/T); % eq 9c
    
end

%%%%%%%%%%%%%%%%%%%
% Output
gradientQ = [dQ_db(2:end), dQ_dg(2:end), dQ_dxi(2:end)]'; % interested only in k=2...K
g=g(2:end); b=b(2:end); xi=xi(2:end);

assert(not(any(isnan(g),'all')))
assert(not(any(isinf(g),'all')))

