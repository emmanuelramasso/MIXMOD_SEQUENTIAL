function [X,current_cluster,pitk] = data_generation(modele,temps,segments)
% segments = [1*Ti, 1*Ti, 3*Ti, 1*Ti] => one segment of length Ti, 1 of
% Ti, one of 3Ti and one of Ti. If Ti=100 => data have 600 lines.
% Timestamps are irregularly spaced, randomly sampled from uniform

[K,dim,~] = size(modele.sigma);
nbdata = sum(segments);

fprintf('Generate %d data...',nbdata);
X = zeros(nbdata,dim);
pitk = zeros(nbdata,K);
current_cluster = zeros(nbdata,1);
    
for t=1:nbdata
    alpha = zeros(1,K);
    alpha(1) = 1;% property eq 4
    for k=2:K
        alpha(k) = modele.beta(k) ./ (1.0 + exp( -modele.gamma(k) .* (temps(t) - modele.tau(k)) ) );% comput alpha
    end
    pitk(t,:) = alpha ./ (sum(alpha, 2)*ones(1,K));% compute pi
    
    % which cluster is activated at t ?
    current_cluster(t) = find(rand < cumsum(pitk(t,:)),1,'first');
    
    % generate the feature vector accordingly
    if not(modele.useExternalFunctions)
        X(t,:) = mvnrnd(modele.mu(current_cluster(t),:),squeeze(modele.sigma(current_cluster(t),:,:)),1);
    else % EXTERNAL
        X(t,:) = mixgauss_sample(modele.mu(current_cluster(t),:)', squeeze(modele.sigma(current_cluster(t),:,:)), ones(1,K), 1)';
    end
end
% figure,plot(pitk)
%figure,plot(X(:,1),X(:,2),'.')
%figure,gscatter(X(:,1),X(:,2),current_cluster)

        