function [onsets,pb] = findOnsets(p,horizAxis,level)
% Find onsets of each cluster in partition p
% p is a vector made of clusters labels
% horizAxis is the vector used to represent the clusters
% for example the time, load, etc. 
% horizAxis must be monotically increasing. 
% level = 1 by default => find cumsum == 1 to define onsets
% otherwise 
% pb=true if one cluster has level < level in the partition
% Emmanuel.Ramasso@femto-st.fr
%

if nargin==2, level = 1;
end

try 
    assert(sum(abs(sort(horizAxis,'ascend')-horizAxis))< eps);
catch 
   error('HorizAxis must be monotically increasing');
end

labels = unique(p); % the different cluster labels
Q = length(labels); % number of clusters

% reorder the clusters - coassociation step
[~,clustersCSCA] = reordonne_clusters(p, Q, 'logCSCA');
if level > 1, clustersCSCA = cumsum(clustersCSCA,1); end
    
onsets = zeros(1,Q);
pb = false; 

for k=1:Q

    f = find(clustersCSCA(:,k)==level,1,'first');
    if isempty(f) % decrease level to minimum
        warning(sprintf('Can not reach this level for k=%d (Q=%d)',k,Q));
        m = max(clustersCSCA(:,k));
        f = find(clustersCSCA(:,k) == m,1,'first');
        disp('Take max = '); disp(m);
        pb = true;
    end
    onsets(k) = horizAxis(f(1));
    
end
onsets = sort(onsets,'ascend');
