% This code runs GMMseq on real data considering differents campaigns. 
% It is not a "generic" code -> to be adapted for your case!
% Can reproduce the results of paper [1] on real data, in particular the
% histograms of onsets. For reproducing the figures on ARI with comparison
% to kmeans, GMM and linkage, please have a look on "comparison.m".
%
%   [1] Emmanuel Ramasso, Thierry Denoeux, Gael Chevallier, Clustering 
%   acoustic emission data stream with sequentially appearing clusters 
%   using mixture models, Mechanical Systems and Signal Processing, 2021.
%
%   [2] Benoit Verdin, Gaël Chevallier, Emmanuel Ramasso, "ORION-AE: 
%   Multisensor acoustic emission datasets reflecting supervised 
%   untightening of bolts in a jointed structure under vibration", 
%   DOI: https://doi.org/10.7910/DVN/FBRDU0, Harvard Dataverse, 2021.
%
%
% Emmanuel Ramasso and Thierry Denoeux
% emmanuel.ramasso@femto-st.fr
% April 2021
%

clear all

saveName = input('Save name ? : ')
dataCampaign = input('Data B C D E F ? : ')

%addpath '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/GITHUB/clustering_fusion/utils'
addpath '/home/emmanuel.ramasso/Documents/CODES/MATLAB/clustering_fusion/utils'
addpath 'netlab3.3'

%%%%%VERIFIER DOSSIER!% c ='/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram/';
%c = '/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram';
c = '/home/emmanuel.ramasso/Documents/CODES/MATLAB';

switch dataCampaign
    case 'B'
            n = 'mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
            m = 'GMMSEQ_B_trustregion_tauprior_lambda1000.mat'
    case 'C'
            n = 'mesure_C_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
            m = 'GMMSEQ_C_trustregion_tauprior_lambda1000.mat'
    case 'D'        
            n = 'mesure_D_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
            m = 'GMMSEQ_D_trustregion_tauprior_lambda1000.mat'
    case 'E'
            n = 'mesure_E_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
            m = 'GMMSEQ_E_trustregion_tauprior_lambda1000.mat'
    case 'F'
            n = 'mesure_F_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
            m = 'GMMSEQ_F_trustregion_tauprior_lambda1000.mat'
    otherwise error('??')
end

% load data
[Xtrain,Ytrain,temps,listFeatures,lesduree,nbFeatInit] = load_data_irt(c, n, false);
% Xtrain = zscore(Xtrain(:,1:length(listFeatures))); % SANS INTERACTION
[Xtrain, mD, sD] = zscore(Xtrain);
labelsTrain = unique(Ytrain);


% in case pca raises an error it means you do not have the statistical
% toolbox. In this case adapt the code with your own pca code.
[coeff, score, latent, tsquared, explained] = pca(Xtrain);
f = find(cumsum(explained)>99,1,'first');
X = score(:,1:f);
figure,g=gscatter(X(:,1),X(:,2),Ytrain); title('PCA1-2')
set(gca,'fontsize',22)
for i=1:length(g), set(g(i),'Marker','s'), end
xlabel('PC1'),ylabel('PC2')
plotmatrix_mine(X(:,1:min(10,size(X,2))),Ytrain)

disp('Size of X')
disp(size(X))

nessais = 10;% 10 with initmodel.minFunc.options.MaxIterations=1500;
sharedCovariances = false;
takePrior = false; % true if you want to explore how the model behaves with prior on onsets
if takePrior
    if dataCampaign == 'C' % for campaign C there is one element missing and set to 0, we remove it
        Kall = 6
        lesduree(lesduree==0) = []; 
    else
        Kall = 7 % nb of clusters
    end
else
    Kall = 4:14 % nb of clusters
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PERFORM CLUSTERING 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meth = {'segments','gmm','kmeans'}; % methods for initialisation
modelesGMMseq = cell(nessais,length(Kall),length(meth));

% we try it possibl'y several times
for essai = 1:nessais
    
    % vary the number of clusters
    iter_K = 0;
    for K=Kall
        
        iter_K = iter_K + 1;
        close all
        
        if takePrior
            penalisation.type = 'l2';
            %penalisation.tauprior = linspace(0,(K-1)/K,K)*(max(temps)-min(temps))+min(temps) % linearly spaced
            penalisation.tauprior = [0 cumsum(lesduree(1:end-1))] % true durations
            penalisation.lambda = 1000;
        end
        
        % vary the initialisation methods
        parfor s = 1:length(meth) 
            
            % for each method some issues (numerical in GMM particularly)
            % can happen, just repeat the initialisation until convergence
            % if it happens. Option matlab-quasinewton often raises an
            % error, sometimes after repetitions we can converge. Otherwise
            % it uses trust-region.
            cont = true;
            it = 1;
            nbitmax = 15; % must be >= 3
            %optimMethod = "matlab-quasinewton"; % best results overall with this method
            %optimMethod = "schmidt";
            %optimMethod="linesearch";
            optimMethod = "matlab-explicitgrad"; % best results overall with this method
            useMiniBatches = true;


            while cont && it <= nbitmax
                
                try                     
                    
                    initmodel = GMMSEQ_init(X,temps,K,meth{s},optimMethod,useMiniBatches,sharedCovariances);
                    disp(initmodel.minFunc.name)
                    
                    if takePrior
                        initmodel.penalisation = penalisation;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % GMMSEQ
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    modele_GMM = GMMSEQ_train(X, temps, initmodel);
                    p = GMMSEQ_test(modele_GMM,X,temps);
                    modele_GMM.type = 'GMM';
                    [~,clustersEstimated]=max(p,[],2);
                    modele_GMM.partititonEstimated = clustersEstimated;
                    
                    if 0
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
                    end
                    
                    disp('GMM SEQ ARI')
                    [ARI_est,~,~,~,C_est] = valid_RandIndex(Ytrain,clustersEstimated);
                    disp(ARI_est)
                    disp('Initialisation ARI')
                    [ARI_init,~,~,~,C_init] = valid_RandIndex(Ytrain,modele_GMM.initmodel.initialPartition);
                    disp(ARI_init)
                    
                    modele_GMM.estim_ARI = ARI_est;
                    modele_GMM.estim_cooccurMat = C_est;
                    modele_GMM.init_ARI = ARI_init;
                    modele_GMM.init_cooccurMat = C_init;
               
                    if 0
                        figure('Name','GMMSEQ'),subplot(2,1,1),plot(temps,clusters_logcsca(clustersEstimated, K)),title('ESTIME')
                        subplot(2,1,2),plot(temps,clusters_logcsca(initmodel.initialPartition,K)),title('INIT')
                        %saveas(gcf,['figures_tmp' filesep sprintf('GMM_SEQ_%d_%d_%d',essai,iter_K,s)],'png')
                        
                        figure,plotmat(C_est,'r','g',12),title(sprintf('%f (%f init)',ARI_est,ARI_init))
                        %saveas(gcf,['figures_tmp' filesep sprintf('GMM_Cest_%d_%d_%d',essai,iter_K,s)],'png')
                    end
                    
                    modelesGMMseq{essai,iter_K,s} = modele_GMM;                    
                    
                    % if we are here it means it successfully converged
                    cont = false;
                    
                catch ME
                    disp('in catch'); 
                    disp(ME); pause(1);
                    disp(it);
                    % in case quasinewton did not converge use trust-region
                    if it == max(1,nbitmax - 2)
                        warning('Change optim method for this iteration.')
                        useMiniBatches = true;
                        optimMethod = "matlab-explicitgrad"; % best results overall with this method
                    end
                    it = it + 1;       
                    
                end
            end
            if cont == true
                error('A problem occurred (roots error in quasinewton/linesearch ?')
            end
        end
    end
end

save(['GMMSEQ_' saveName])







