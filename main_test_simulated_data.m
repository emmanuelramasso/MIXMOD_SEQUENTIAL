% We initialise a true model, and generate the data
% We run several initialisations of GMMseq and select the best model
% Results are then plotted
% Images can be saved if needed (see boolean below)
% This code allows to reproduce the results of [1] on simulated data.
%
% [1] Emmanuel Ramasso, Thierry Denoeux, Gael Chevallier, Clustering 
%   acoustic emission data stream with sequentially appearing clusters 
%   using mixture models, Mechanical Systems and Signal Processing, 2021.
%
% Emmanuel Ramasso and Thierry Denoeux
% emmanuel.ramasso@femto-st.fr
% April 2021
%

saveImages = false; % true => create images in "tmpImages" directory 
                    % and save some images (of the paper) in current directory

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
s = cell(1,K); for k=1:K, s{k} = sprintf('Cluster %d',k); end
legend(s)
if saveImages
    saveas(gcf,'pik_true','epsc'),
    saveas(gcf,'pik_true','fig'),
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Reproduce some images
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 0
    reproduce_figures_2_and_3;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Training
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% here we fix K, a study vs K is made in another script
% To get a good estimation try "matlab-quasinewton" (tested on 2020b) with
% initmodel.thnorm2 = 1e-3 and initmodel.thresholdLLdiff = 1e-4
K = 4; % known number of clusters
initMethods = {'gmm','kmeans','segments'}; % different types of initalizations
optimMethods = ["matlab-explicitgrad", ... % use a matlab function: explicit gradients and trust region algorithm
    "matlab-quasinewton",  ... % use a matlab function: explicit gradients with Wolfe method and quasi netwon method to estimate Hessian
    "linesearch",...           % use external function for a linesearch
    "schmidt"];                % use LBFGS + Wolfe linesearch
optimMethod = optimMethods(2)              % most of optim methods work with those simulated datasets
sharedCovariances = false; % share or not the variances between clusters
bestLL = -inf;
nbtrials = 10;
for l=1:length(initMethods)
    initMethod = initMethods{l};
    fprintf('Method: %s\n',initMethod);
    fprintf('Nb trials = %d:',nbtrials);
    
    for i=1:nbtrials % makes several runs and select the model with the highest likelihood
        fprintf(' %d ',i);
        
        cont = true; it = 1;
        while cont
            
            useMiniBatches = size(X,1) > 2000 && lower(optimMethod)=="matlab-explicitgrad"; % evaluate gradients on all data, advised for more than ~2000 data
            
            initmodel = GMMSEQ_init(X,temps,K,initMethod,optimMethod,useMiniBatches,sharedCovariances);
            
            % modify some thresholds to wait for finer results
            initmodel.thnorm2 = 1e-3;         % threshold for ||xnew-xold|| = norm(xnew - xold);
            initmodel.thresholdLLdiff = 1e-4; % threshold on the evolution of the likelihood
            
            % Sometimes, when using matlab-quasi newton we have the following
            %error : "Error using roots (line 27) Input to ROOTS must not
            %contain NaN or Inf.". Just rerun the algorithm a few times.
            try
                m = GMMSEQ_train(X, temps, initmodel);
                assert(m.convergenceProblem == false); % else go in catch
                cont = false;% here GMMSEQ succeeded to find a model
            catch
                it = it + 1;
                fprintf('An error occurred, retry... (%d/%d)',it,10);
                if it >= 10, error('Cannot converge ??'); end
            end
        end
        
        if m.loglik > bestLL
            modele_GMM = m; % keep model
            bestLL = m.loglik;
        end
    end
    fprintf('\n');
    
end
fprintf('Done.\n');

% % In case of regularisation
% penalisation.type = 'l2';
% penalisation.tauprior = true_model.tau;
% % Example of priors
% penalisation.tauprior = linspace(0,(K-1)/K,K)*(max(temps)-min(temps))+min(temps)
% penalisation.lambda = 1; % > 0
% initmodel.penalisation = penalisation;
% % then run the model


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% check the likelihood function (figure 4)
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you increase the thresholds initmodel.thnorm2 and/or
% initmodel.thresholdLLdiff then the model will be less precise
% and far from the maximum likelihood
if 0
    meshstep = 200; % the bigger the slower but the finer in terms of scale
    % can be checked for every cluster
    for clusterChoisi = 2:K % > 1 because cluster 1 has neither beta nor tau
        %test_check_likelihood_contour(true_model,X,temps,clusterChoisi,meshstep);
        test_check_likelihood_contour(modele_GMM,X,temps,clusterChoisi,meshstep,saveImages);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Plots
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get the activations and clusters
[p,pitk_est] = GMMSEQ_test(modele_GMM,X,temps);
[~,clustersEstimated] = max(p,[],2);

figure('name','GMM with sequence');
subplot(131), hold on, title('Initialisation','interpreter','latex','Fontsize',24); set(gca,'fontsize',22);
xlabel('$X_1$','interpreter','latex'); ylabel('$X_2$','interpreter','latex')
subplot(132), hold on, title('Estimated','interpreter','latex','Fontsize',24); set(gca,'fontsize',22)
xlabel('$X_1$','interpreter','latex'); ylabel('$X_2$','interpreter','latex')
subplot(133), hold on, title('True','interpreter','latex','Fontsize',24); set(gca,'fontsize',22)
xlabel('$X_1$','interpreter','latex'); ylabel('$X_2$','interpreter','latex')
for i=1:modele_GMM.initmodel.nb_clusters
    % initialisation
    pp = modele_GMM.initmodel.initialPartition == i;
    subplot(131),plot(X(pp,1),X(pp,2),'*')
    % estimated
    pp = clustersEstimated==i;
    subplot(132),plot(X(pp,1),X(pp,2),'*')
    % true
    pp = current_cluster == i;
    subplot(133),plot(X(pp,1),X(pp,2),'*')
end
set(gcf,'Position',[1           1        1920         954])
% if saveImages
% saveas(gcf,'simu_X1X2_result_clustering_GMMseq','fig')
% saveas(gcf,'simu_X1X2_result_clustering_GMMseq','epsc')
% end

figure,plot(modele_GMM.history_loglik,'linewidth',3),
ylabel('Log-likelihood of GMMSEQ','interpreter','latex','Fontsize',24),
xlabel('iteration','interpreter','latex','Fontsize',24)
set(gca,'fontsize',22); grid minor
set(gcf,'Position',[1           1        1920         954])
% if saveImages
% saveas(gcf,'simu_X1X2_loglik','fig')
% saveas(gcf,'simu_X1X2_loglik','epsc')
% end

disp('GMM SEQ ARI')
[ARI_est,~,~,~,C_est] = valid_RandIndex(current_cluster,clustersEstimated);
disp(ARI_est)
[ARI_init,~,~,~,C_init] = valid_RandIndex(current_cluster,modele_GMM.initmodel.initialPartition);
disp('Initialisation ARI')
disp(ARI_init)

disp('tau vrais vs estimés')
true_model.tau
modele_GMM.tau
disp('beta vrais vs estimés')
true_model.beta
modele_GMM.beta
disp('gamma vrais vs estimés')
true_model.gamma
modele_GMM.gamma


[p,pitk_init] = GMMSEQ_test(modele_GMM.initmodel,X,temps);
figure,plot(temps,pitk_init),title('\pi_{tk} initialization')
figure,plot(temps,pitk_est),title('\pi_{tk} estimated')

figure,plot(temps,pitk_est,'linewidth',2), %title('\pi_{tk} - Estimated')
set(gca,'Fontsize',30)
set(gcf,'Position',[1           1        1920         954])
ylabel('\pi_{tk} (estimated)')
xlabel('Time')
s = {}; for k=1:K, s{end+1} = sprintf('Cluster %d',k); end
legend(s)
if saveImages
    saveas(gcf,'pik_est','epsc'),
    saveas(gcf,'pik_est','fig'),
end

figure,plot(temps,pitk_init,'linewidth',2), %title('\pi_{tk} - Initialization')
set(gca,'Fontsize',30)
set(gcf,'Position',[1           1        1920         954])
ylabel('\pi_{tk} (initialization)')
xlabel('Time')
s = {}; for k=1:K, s{end+1} = sprintf('Cluster %d',k); end
legend(s)
if saveImages
    saveas(gcf,'pik_init','epsc'),
    saveas(gcf,'pik_init','fig'),
end












