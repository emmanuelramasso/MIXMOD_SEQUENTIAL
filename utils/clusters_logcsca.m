function logcumc = clusters_logcsca(clusters, Q)
% clusters = partitions
% Q = number of clusters

nbpts = numel(clusters);
logcumc = zeros(Q,nbpts);
logcumc(clusters(:)' + [0:Q:Q*nbpts-1]) = 1;
logcumc = log10(cumsum(logcumc,2));
logcumc = logcumc';