% This code runs GMMseq on real data considering differents campaigns
% It is not a "generic" code -> to be adapted for your case!
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
%


clear all

dataCampaign = input('Data B C D E F ? : ')

%addpath '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/GITHUB/clustering_fusion/utils'
addpath '/home/emmanuel.ramasso/Documents/CODES/MATLAB/clustering_fusion/utils'
addpath 'netlab3.3'

%%%%%VERIFIER DOSSIER!% c ='/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram/';
c = '/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram';

switch dataCampaign
    case 'B'
        n = 'mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
        
    case 'C'
            n = 'mesure_C_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'

    case 'D'        
            n = 'mesure_D_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'

    case 'E'
            n = 'mesure_E_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
    
    case 'F'
            n = 'mesure_F_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
    
    otherwise error('??')
end

% load data
[Xtrain,Ytrain,temps,listFeatures,lesduree,nbFeatInit] = load_data_irt(c, n, false);
% Xtrain = zscore(Xtrain(:,1:length(listFeatures))); % SANS INTERACTION
[Xtrain, mD, sD] = zscore(Xtrain);
labelsTrain = unique(Ytrain);

% PCA
[coeff, score, latent, tsquared, explained] = pca(Xtrain);
f = find(cumsum(explained)>99,1,'first');
X = score(:,1:f);
figure,g=gscatter(X(:,1),X(:,2),Ytrain); title('PCA1-2')
set(gca,'fontsize',22)
for i=1:length(g), set(g(i),'Marker','s'), end
xlabel('PC1'),ylabel('PC2')
% plotmatrix_mine(X(:,1:min(10,size(X,2))),Ytrain)

disp('Size of X')
size(X)

K = 7
sharedCovariances = false;
optimMethod = "matlab-explicitgrad";%"schmidt"; %"matlab-quasinewton"; %"trust-region"; % best results overall with this method
initMethod = 'gmm';
useMiniBatches = size(X,1) > 2000 && lower(optimMethod)=="matlab-explicitgrad"; 

% init model GMMseq
initmodel = GMMSEQ_init(X,temps,K,initMethod,optimMethod,useMiniBatches,sharedCovariances);
initmodel.minFunc.options.MaxIterations = 1;
initmodel.useMiniBatches = true; % dependence to sampling

% in case 
%if 0
%    penalisation.type = 'l2';
%    penalisation.tauprior = linspace(0,(K-1)/K,K)*(max(temps)-min(temps))+min(temps)
%penalisation.tauprior = zeros(1,K);
%    penalisation.lambda = 1000;
%    initmodel.penalisation = penalisation;
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMMSEQ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

modele_GMM = GMMSEQ_train(X, temps, initmodel);
p = GMMSEQ_test(modele_GMM,X,temps);
modele_GMM.type = 'GMM';
[~,clustersEstimated]=max(p,[],2);
modele_GMM.partititonEstimated = clustersEstimated;

figure('name','GMM with sequence')
subplot(131), hold on, title('Initialisation')
subplot(132), hold on, title('Estimated')
subplot(133), hold on, title('True')
for i=1:initmodel.nb_clusters
    % initialisation
    pp = modele_GMM.initmodel.initialPartition == i;
    subplot(131),plot(X(pp,1),X(pp,2),'*')
    % estimated
    pp = clustersEstimated==i;
    subplot(132),plot(X(pp,1),X(pp,2),'*')
end
% true (more or less clusters than initmodel.nb_clusters
for i=1:length(labelsTrain)
    pp = Ytrain == i;
    subplot(133),plot(X(pp,1),X(pp,2),'*')
end

disp('GMM SEQ ARI')
[ARI_est,~,~,~,C_est] = valid_RandIndex(Ytrain,clustersEstimated)
disp('Initialisation ARI')
[ARI_init,~,~,~,C_init] = valid_RandIndex(Ytrain,modele_GMM.initmodel.initialPartition)

figure('Name','GMMSEQ'),subplot(2,1,1),plot(temps,clusters_logcsca(clustersEstimated, K)),title('ESTIME')
subplot(2,1,2),plot(temps,clusters_logcsca(initmodel.initialPartition,K)),title('INIT')
%saveas(gcf,['figures_tmp' filesep sprintf('GMM_SEQ_%d_%d_%d',essai,iter_K,s)],'png')

figure,plotmat(C_est,'r','g',12),title(sprintf('%f (%f init)',ARI_est,ARI_init))
%saveas(gcf,['figures_tmp' filesep sprintf('GMM_Cest_%d_%d_%d',essai,iter_K,s)],'png')






