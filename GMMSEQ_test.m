function [p,pi] = GMMSEQ_test(modele,data,time)
%[p,pi] = GMMSEQ_test(modele,data,time)
% Returns the posterior probability on each class given data and a GMMSEQ
% model. Inputs follows the same nomenclature as GMMSEQ_train.m.
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



x = setXfromParam(modele);

[~,p,~,pi] = GMMSEQ_loglikelihood(x, modele.mu, modele.sigma, modele.nb_clusters, modele.T, data, time);
