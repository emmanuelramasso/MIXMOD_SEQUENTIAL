% test on simulated data
% program close to main_test_simulated_data.m
% except that run GMMSEQ using different initialisations
% get the best models for each number of clusters and plot the criteria for selection
% close to "main_simulated_data.m"
% About 4h to run in the current state
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


clear all
close all


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% data with K=4 clusters
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = 4;
mu1 = [1 1]; Sigma1 = [.3 .2; .2 .2];
mu2 = [2 3]; Sigma2 = [.3 .2; .2 .2];
mu3 = [3 5]; Sigma3 = [.2 .1; .1 .3];
mu4 = [5 6]; Sigma4 = [.2 .1; .1 .2];
true_model.mu = [mu1; mu2; mu3; mu4];
true_model.sigma(1,:,:) = Sigma1; true_model.sigma(3,:,:) = Sigma3;
true_model.sigma(2,:,:) = Sigma2; true_model.sigma(4,:,:) = Sigma4;

Ti = 1000;% length of data per cluster
groundTruth = [1*Ti, 1*Ti, 3*Ti, 1*Ti];
L = sum(groundTruth);
temps = cumsum(rand(L,1)); % random Delta time between events
% temps = (1:L)'; % regular Delta time between events
T = max(temps);

% true model
true_model.L = L;
true_model.T = T;
true_model.nb_clusters = K;
true_model.tau = [nan, temps(cumsum(groundTruth(2:end)))'];
true_model.gamma = [nan, 0.9 1.5 1.2]/100;
true_model.beta = [nan, cumsum(exp(1:K-1))]; 

try % STATS TOOLBOX AVAILABLE ?
    mvnpdf(0,1,1);
    true_model.useExternalFunctions = false;
catch 
    true_model.useExternalFunctions = true;
end

% generate data
[X,current_cluster,pitk] = data_generation(true_model,temps,groundTruth); 

fprintf('ok.\nNb of points per cluster:\n');
if not(true_model.useExternalFunctions)
    tabulate(current_cluster)
end

figure,plot(temps,current_cluster),title('True clusters')

figure,plot(temps,pitk,'linewidth',2), %title('\pi_{tk} - True model')
set(gca,'Fontsize',30)
set(gcf,'Position',[1           1        1920         954])
ylabel('\pi_{tk} (true model)')
xlabel('Time')
s = {}; for k=1:K, s{end+1} = sprintf('Cluster %d',k); end
legend(s)
saveas(gcf,'pik_true','epsc'), 
saveas(gcf,'pik_true','fig'), 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% check the likelihood function
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

if 0    
    % on choisit un cluster (entre 2 et K)
    clusterChoisi = 2; % doit etre > 1
    % on appelle la portion de code qui va faire varier un paramètre beta dans
    % ce cluster et le paramètre tau dans ce cluster, puis évaluer la fonction
    % de vraisemblance.
    test_check_likelihood_contour(true_model,X,temps,clusterChoisi);    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in case of regularisation
% penalisation.type = 'l2';
% penalisation.tauprior = true_model.tau;
% %penalisation.tauprior = linspace(0,(K-1)/K,K)*(max(temps)-min(temps))+min(temps)
% penalisation.lambda = 1; % > 0 
% initmodel.penalisation = penalisation;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% model estimation
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 4 expected, try between 2 and 10
lesK = 2:10;
initMethods = {'gmm','kmeans','segments'};
sharedCovariances = false; % share or not the variances between clusters
useMiniBatches = true; % evaluate gradients on all data, advised for more than 2000 data
optimMethod = "matlab-explicitgrad"; % use explicit gradients

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM SEQ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

timetests = tic;
close all; k=1;
gmmseq = cell(1,length(lesK));

for l=1:length(initMethods)
    
    initMethod = initMethods{l};
    
    for K=lesK % vary nb clusters
        
        for itest=1:10 % make 10 trials
            
            try
                
                close all
                
                initmodel = GMMSEQ_init(X,temps,K,initMethod,optimMethod,useMiniBatches,sharedCovariances);
                
                initmodel_GMM = initmodel; %initmodel
                modele_GMM = GMMSEQ_train(X, temps, initmodel_GMM)
                p = GMMSEQ_test(modele_GMM,X,temps);
                [~,clustersEstimated] = max(p,[],2);
                
                % figure('name','GMM with sequence')
                % subplot(131), hold on, title('Initialisation')
                % subplot(132), hold on, title('Estimated')
                % subplot(133), hold on, title('True')
                % for i=1:initmodel.nb_clusters
                %     % initialisation
                %     pp = initmodel.initialPartition == i;
                %     subplot(131),plot(X(pp,1),X(pp,2),'*')
                %     % estimated
                %     pp = clustersEstimated==i;
                %     subplot(132),plot(X(pp,1),X(pp,2),'*')
                %     % true
                %     pp = current_cluster == i;
                %     subplot(133),plot(X(pp,1),X(pp,2),'*')
                % end
                
                % figure,plot(modele_GMM.history_loglik),ylabel('Likelihood GMM with sequence'),xlabel('iteration')
                
                disp('GMM SEQ ARI')
                [ARI_est,~,~,~,C_est] = valid_RandIndex(current_cluster,clustersEstimated);
                disp(ARI_est)
                [ARI_init,~,~,~,C_init] = valid_RandIndex(current_cluster,modele_GMM.initmodel.initialPartition);
                disp('Initialisation ARI')
                disp(ARI_init)
                
                modele_GMM.ARIest  = ARI_est;
                modele_GMM.ARIinit = ARI_init;
                
                gmmseq{k} = modele_GMM; % STORE ALL
            catch
                gmmseq{k} = [];
                '??'
            end
            k=k+1;
            
        end
        
    end
    
end
toc(timetests)

% save essai_simu_init_gmm_et_kmeans_sharedCov
% save essai_simu_init_gmm_et_kmeans_CovFull

% store nb clusters and logliks
LL = -inf(length(gmmseq),1);
lesK = -inf(length(gmmseq),1);
for i=1:length(gmmseq)
    if ~isempty(gmmseq{i})
        LL(i) = gmmseq{i}.loglik;
        lesK(i) = gmmseq{i}.nb_clusters;
    else
        LL(i) = -inf;
        lesK(i) = -inf;
    end
end

% select the likeliest model for each number of clusters
u = unique(lesK); u(isinf(u)) = [];
p = zeros(length(u),1);
clear bestModel
for i=1:length(u)
    f = find(lesK==u(i));
    [ll,a] = max(LL(f));
    p(i) = f(a);
    bestModel(i) = gmmseq{p(i)};
end

% Plot the criteria for each cluster number
figure, hold on
set(gcf,'Position',[1           1        1920         954])
set(gca,'Fontsize',28)
xlabel('Nb clusters'), ylabel('Criteria'), grid minor 
for i=1:length(bestModel)
    plot(bestModel(i).nb_clusters,bestModel(i).criteria.AIC,'bo','linewidth',3,'markersize',12)
    plot(bestModel(i).nb_clusters,bestModel(i).criteria.BIC,'rx','linewidth',3,'markersize',12)
    plot(bestModel(i).nb_clusters,bestModel(i).criteria.ICL,'g^','linewidth',3,'markersize',12)
end
legend('AIC','BIC','ICL')
saveas(gcf,'criteria_simu','epsc')
saveas(gcf,'criteria_simu','fig')

% disp the ARI
for i=1:length(bestModel)
   [bestModel(i).nb_clusters,bestModel(i).ARIest]
end

% Plot the activation function
m = bestModel(3) % K=4 clusters

[p,pitk_est] = GMMSEQ_test(m,X,temps);
[p,pitk_init] = GMMSEQ_test(m.initmodel,X,temps);
[p,pitk_true] = GMMSEQ_test(true_model,X,temps);


figure,plot(temps,pik1,'linewidth',2)
ylabel('$\pi_k$','interpreter','latex','fontsize',32),
xlabel('Time','interpreter','latex','fontsize',32),
set(gca,'Fontsize',30)
set(gcf,'Position',[1           1        1920         954])
drawnow
saveas(gcf,'pik_true','epsc'), 
saveas(gcf,'pik_true','fig'), 


figure,plot(temps,pitk_est,'linewidth',2), %title('\pi_{tk} - Estimated')
set(gca,'Fontsize',30)
set(gcf,'Position',[1           1        1920         954])
ylabel('\pi_{tk} (estimated)')
xlabel('Time')
s = {}; for k=1:K, s{end+1} = sprintf('Cluster %d',k); end
legend(s)
saveas(gcf,'pik_est','epsc'), 
saveas(gcf,'pik_est','fig'), 

figure,plot(temps,pitk_init,'linewidth',2), %title('\pi_{tk} - Initialization')
set(gca,'Fontsize',30)
set(gcf,'Position',[1           1        1920         954])
ylabel('\pi_{tk} (initialization)')
xlabel('Time')
s = {}; for k=1:K, s{end+1} = sprintf('Cluster %d',k); end
legend(s)
saveas(gcf,'pik_init','epsc'), 
saveas(gcf,'pik_init','fig'), 

figure,plot(temps,pitk_true,'linewidth',2), %title('\pi_{tk} - True model')
set(gca,'Fontsize',30)
set(gcf,'Position',[1           1        1920         954])
ylabel('\pi_{tk} (true model)')
xlabel('Time')
s = {}; for k=1:K, s{end+1} = sprintf('Cluster %d',k); end
legend(s)
saveas(gcf,'pik_true','epsc'), 

% save essai_simu_init_gmm_sharedCov
% save essai_simu_init_kmeans_sharedCov







