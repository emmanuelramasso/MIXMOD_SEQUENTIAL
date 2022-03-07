function modele = GMMSEQ_train(data, time, initmodel)
%GMMSEQ_train    Find the ML estimates of a multivariate mixture
% distribution of Gaussians as well as optimal parameters of sigmoids in 
% the GMMSEQ model. The structure initmodel, created by GMMSEQ_init.m,
% allows to set on fields to "generic values". Some of them should be set
% according to your own datasets such as the number of iterations, the
% thresholds on the likelihood (convergence), the use or not of shared
% covariance, the type of minimizer (optimization), the verbosity (useful 
% to understand some details), and so on. An example for the use of this 
% code is provided in main_test_simulated_data.m.
%
%    INPUTS:
%       data is a matrix with L lines, dx columns
%       time is a vector of L elements 
%       initmodel is initialised with GMMSEQ_init.m function.
%
%    OUTPUTS:      
%       modele.mu: means of Gaussians 
%       modele.sigma: covariances of Gaussians 
%       modele.beta: amplitude (cumulated value) of sigmoids 2...K
%       modele.gamma: gamma (slopes) value of sigmoids 2...K
%       modele.tau: tau (onsets) value of sigmoids 2...K 
%       modele.x: beta, gamma and tau expressed with instrumental variables
%       modele.loglik: loglikelihood of the model
%       modele.history_loglik: history of likelihood
%       modele.nb_clusters: number of clusters
%       modele.T: max(time) if time provided
%       modele.L: size of data
%       modele.initmodel: initial model
%       modele.nparams: number of parameters in the model
%       modele.criteria: AIC, BIC and ICL for the model 
%       modele.convergenceProblem: indicates whether a pb occurs
%
%   NOTE
%
%   This code makes use of several toolboxes from Matlab, in particular the
%   optimization toolbox and statistical toolbox. Some functions from other
%   people are included in the package in case you do not have those
%   toolboxes. For example minFunc from Schmidt and BayesNet from Murphy
%   for optimizing and computing some distributions respectively. In
%   practice we used the toolboxes of Mathworks which seems to have better
%   stability according to numerical issues. Results shown in the paper
%   have been obtained using Mathworks toolboxes, in particular the 
%   trust-region algorithm. Some numerical issues can still occur, some are
%   detected in the code and in that case a variable "convergenceProblem"
%   is set to 1 in the output structure. 
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



%%%%%%%%%%%%%%%%%%%%
% check format
K = initmodel.nb_clusters;
[L,dx] = size(data);
assert(size(time,1)==L);
assert(all(diff(time)>=0))
T = max(time);
assert(isfield(initmodel,'gamma') & isfield(initmodel,'beta') & isfield(initmodel,'tau'));
muold = initmodel.mu;
sigmaold = initmodel.sigma;
assert( all( size(muold) == [K,dx] ) );
if dx>1, assert( all( size(sigmaold) == [K,dx,dx] ) );
else, assert( all( size(sigmaold) == [K,dx] ) );
end
modele.T = T;
modele.L = L;
modele.convergenceProblem = false;
initmodel.sizeMiniBatches = min(L,initmodel.sizeMiniBatches);

if initmodel.sharedCovariances
    m = squeeze(mean(sigmaold,1));
    for k=1:K
        sigmaold(k,:,:) = m;
    end
end

% get param of sigmoids with instrumental variables
xold = setXfromParam(initmodel);

% set some variables for iterations to come
iter = 1;
LL0 = -inf;
histLL = zeros(initmodel.nitermax,1); 
histLL(1) = LL0;
loopagain = true;
cases = 0; 
maxCases = 0;
TrepK = time*ones(1,K, 'double');% replicated time instants for 1...K
stored_norm = zeros(initmodel.nitermax,1); 
stored_crit_forConvergenceCheck = zeros(initmodel.nitermax,2); 

fprintf('GMMSEQ training...\n')
totaltime = tic;
while loopagain
    
    if initmodel.verbosity
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        fprintf('Iteration %d (K=%d)\n',iter,K)
    end
    
    %%%%%%%%%%%%%%%%%%%%
    % E-STEP          
    %%%%%%%%%%%%%%%%%%%%
    
    if iter == 1
        [~, ~, ~, yold] = Estep(xold, K, data, TrepK, T, muold, sigmaold, initmodel.useExternalFunctions);
    else
        % it will be computed by [LLnew,ynew,alphanew,pinew] = GMMSEQ_loglikelihood(xnew, munew, sigmanew, K, T, data, TrepK, initmodel.useExternalFunctions);
        % for checking convergence afterwards so we do not need to compute it again
    end
    
    %%%%%%%%%%%%%%%%%%%%
    % MSTEP
    %%%%%%%%%%%%%%%%%%%%
    if initmodel.verbosity
        fprintf('M-step for means and cov...')
    end
  
    % -----------------------------------------------------
    % 1. Covariance and means 
    % optim
    s = sum(yold, 1);% eq 9.27 CB
    sigmanew = zeros(K, dx, dx, 'double');
    munew = zeros(K, dx, 'double');  
    vec = false(1,K);
    
    for k=1:K                    
        
        if all(yold(:, k) == 0) 
            %k, error('Pb for this cluster : all probs are == 0 ==> decrease K ?'); 
            vec(k) = true;
        end
        
        munew(k, :) = sum( data .* yold(:, k) ) / s(k);% Means / eq 9.24 CB        
        
        diffs = data - munew(k, :); %  x-mu 
        diffs = diffs .* sqrt(yold(:,k));% for cov computation
        sigmanew(k,:,:) = (diffs'*diffs) / s(k);% Cov / eq 9.25 CB   
                
        if not(initmodel.sharedCovariances) && not(vec(k)) && min(svd(squeeze(sigmanew(k,:,:)))) < 1e-10 % Ensure that covariances don't collapse with minimum singular value of covariance matrix
           sigmanew(k,:,:) = sigmaold(k,:,:); 
           if initmodel.verbosity
               warning('I use previous estimates for covMat of cluster %d - sv', k);
           end
        end
    end
    
    % delete clusters if empty
    if any(vec)
        if initmodel.verbosity
            fprintf('REMOVE CLUSTER %s',num2str(find(vec)))
        end
        modele.convergenceProblem = true;
        %assert(not(vec(1)==1))
        x2 = reshape(xold,[],K-1);
        x2 = [nan(size(x2,1),1), x2];
        x2(:,find(vec)) = [];
        x2(:,1) = [];
        xold = reshape(x2,1,[]);
        K = K - sum(double(vec)); % new nb of clusters
        yold(:,vec) = [];
        TrepK(:,vec) = [];
        sigmanew(vec,:,:) = [];
        munew(vec,:) = [];
        sigmaold(vec,:,:) = [];
        muold(vec,:,:) = [];
    end
    
    if initmodel.sharedCovariances
        m = squeeze(mean(sigmanew,1));
        for k=1:K
            sigmanew(k,:,:) = m;
        end
    end
    
    % check if the matrix has a proper format 
    [err, munew, sigmanew, kpb] = check_posdef(munew, sigmanew, muold, sigmaold);
    if err==0, if initmodel.verbosity, fprintf('ok.\n');  end
    else
        modele.convergenceProblem = true;
        modele.convergenceProblemMessage = '[1] An error occurs (not pos def cluster %d), corrections were made.';
        if initmodel.verbosity
            fprintf('\n-->[1] An error occurs (not pos def cluster %d), corrections were made.\n',kpb)
        end
    end    
    
    % -----------------------------------------------------
    % 2. Gamma, Beta and Tau
    if initmodel.verbosity
        fprintf('Optimising gamma, tau and beta...\n')
    end
    
    % Initialise with previous estimate and apply one optimiser
    % Matlab fminunc generally gives better estimates but IS LONG LONG! for K high        
    
    problem.x0 = xold(:)'; % start with previous parameters
    problem.objective = @(x) auxiliaryFunction(x, yold, K, T, L, TrepK, initmodel);
    
    if initmodel.minFunc.name == "schmidt"
        
        if initmodel.verbosity
            fprintf('-> use minFunc/Schmidt as minimiser...')        
        end
        xnew = minFunc(problem.objective, problem.x0(:), initmodel.minFunc.options); 
        
    elseif initmodel.minFunc.name == "linesearch"
        
        if initmodel.verbosity
            fprintf('-> use linesearch/Bangert minimiser...')
        end
                
        % version 1
        %xnew = lineSearch_reloaded(problem.x0(:),problem.objective,initmodel.minFunc.options);
        
        % version 2
        %[~, gradientQ] = auxiliaryFunction(xold, yold, K, T, L, TrepK, initmodel);
        %problem.objective = @(delta) auxiliaryFunction(xold+delta*gradientQ, yold, K, T, L, TrepK, initmodel);
        %problem.x1 = 0;
        %problem.x2 = 0.001;
        %problem.solver = 'fminbnd';
        %problem.options = optimset('TolX',eps,'MaxFunEvals',5000,'MaxIter',5000);        
        %alpha          = fminbnd(problem); 
        %xnew = xold(:) + alpha*gradientQ;
        
        % version 3
        [~, gradientQ] = auxiliaryFunction(xold, yold, K, T, L, TrepK, initmodel);
        lesetas = 0:1e-5:0.01;
        QQ = zeros(length(lesetas),1);
        k = 1;
        for eta=lesetas
            xxx = xold + eta * gradientQ;
            QQ(k) = auxiliaryFunction(xxx, yold, K, T, L, TrepK, initmodel);
            k = k + 1;
        end
        %figure,plot(lesetas,QQ)
        [a,b] = min(QQ);
        eta = lesetas(b);
        xnew = xold(:) + eta*gradientQ;
        clear k eta lesetas gradientQ QQ xxx
        
    else
              
        problem.solver = 'fminunc';
        problem.options = initmodel.minFunc.options;        
        if initmodel.verbosity
            fprintf('-> use fminunc/Matlab as minimiser...')
        end
        xnew = fminunc(problem);
        
    end      
    %fprintf('Estimated\t\t Calculé\t Diff\n'); 
    %for i=1:20, fprintf('%f \t %f \t %f\n',grad_fd(i),grad(i),grad_fd(i)-grad(i)); end    
    if initmodel.verbosity
        fprintf('Ok.\n');
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    % check convergence
    [LLnew,ynew,alphanew,pinew] = GMMSEQ_loglikelihood(xnew, munew, sigmanew, K, T, data, TrepK, initmodel.useExternalFunctions);

    % Messages
    if initmodel.verbosity
        disp('LL before'); fprintf('%f\n',LL0);
        disp('LL after');  fprintf('%f\n',LLnew);
    end
    
    histLL(iter) = LLnew;
    %delta_loglik = abs(LLnew - LL0);
    %avg_loglik = (abs(LLnew) + abs(LL0) + eps)/2.0;
    %cond1 = (delta_loglik / avg_loglik) < initmodel.deltaJmintoExit;
    %cond2 = abs(LLnew - LL0) < initmodel.thresholdLLdiff;
    llmov = median(abs(diff(histLL(max(1,iter-5):iter)))); % median diff of loglik between iterations on a small window
    cond2 = llmov < initmodel.thresholdLLdiff;
    cond2b = llmov < initmodel.thresholdLLdiff/10;
    
    cond3 = iter > initmodel.nitermax;   
    
    thetaold = [xold(:); muold(:); sigmaold(:)];    
    thetanew = [xnew(:); munew(:); sigmanew(:)];
    nx2 = norm(thetanew - thetaold);   
    stored_norm(iter) = nx2;
    mnorm = median(stored_norm(max(1,iter-5):iter)); % median norm of parameters between iterations on a small window
    cond5  = mnorm <= initmodel.thnorm2; % threshold 1
    cond5c = mnorm <= initmodel.thnorm2/10;% sometimes likelihood oscillates so the norm will stop the process
    
    % if iter exceeds limits or norm of parameters between two iterations is
    % very low or likelihood does not increase and norm of parameters between 
    % two iterations is low then convergence reached
    if (cond5 && cond2) || cond3 || cond5c || cond2b 
        loopagain = false;
    end    
    if initmodel.verbosity
        fprintf('---------------- Stats convergence ---------------- \n')
        fprintf('AVG( || xnew-xold || ) = %f (th=%f)\n',mnorm,initmodel.thnorm2);
        fprintf('AVG( | LLnew-LLold | ) = %f (th=%1.10f) \n',llmov,initmodel.thresholdLLdiff)
        fprintf('-------------------------------------------- \n')
    end
    
    if LLnew - LL0 < -1e-3 
        fprintf('\n--> !! Likelihood decreases substantially (%d/%d), continue some iterations to confirm.\n',cases,maxCases);
        cases = cases + 1;
        if cases >= maxCases
            % STOP algorithm because of numerical problems (rare in general)
            if initmodel.verbosity
                warning('Convergence pb? (LLold > LLnew) -> get out.')
            end
            modele.convergenceProblem = true; 
            modele.convergenceProblemMessage = 'Convergence pb? (LLold > LLnew) -> get out.';
            pause(1), break;
        end
    else
        if initmodel.verbosity
            disp('Model was improved!')
        end
    end
   
    % store new values
    LL0 = LLnew;
    xold = xnew;
    muold = munew; 
    sigmaold = sigmanew;
    yold = ynew; % avoid the E step
    
    % plot some results
    if initmodel.plotDuringIter
        figure(100),
        if iter==1, clf; end
        subplot(331),plot(time,pinew),title('$\pi_{tk}$','interpreter','latex')
        subplot(334),plot(time,alphanew),,title('$\alpha_{tk}$','interpreter','latex')
        subplot(337),plot(time,ynew),title('$y_{tk}$','interpreter','latex')
        [~,~,~,beta,gamma,tau] = getParamFromX(xnew,K,T);
        subplot(332),hold on, plot(iter,beta,'ko','markersize',5),title('$\beta_{k}$','interpreter','latex')
        subplot(335),hold on, plot(iter,gamma,'gx','markersize',5),title('$\gamma_{k}$','interpreter','latex')
        subplot(338),hold on, plot(iter,tau,'bs','markersize',5),title('$\tau_{k}$','interpreter','latex')
        subplot(3,3,[3 6]),plot(histLL(1:iter-1)),title('Likelihood')
        [~,clustersEstimated] = max(ynew,[],2);
        %%c = visualiser_resultats_shm_clust_morceaux_2(clustersEstimated,time,K,[],0);
        logcumc = clusters_logcsca(clustersEstimated, K);
        subplot(3,3,9),plot(time,logcumc),title('Sequence')
        drawnow
    end
    
    % next iteration
    iter = iter+1;
     
end
totaltime = toc(totaltime);
fprintf('End in %d iter (%f s).\n',iter,totaltime)

if iter >= initmodel.nitermax
    warning('Number of iterations probably too small.')
    modele.convergenceProblem = true;    
    modele.convergenceProblemMessage = 'Number of iterations probably too small.';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
histLL(iter:end)=[];
[~,~,~,modele.beta,modele.gamma,modele.tau] = getParamFromX(xold,K,T);
modele.mu = muold;
modele.sigma = sigmaold;
modele.loglik = LL0;
modele.totaltime = totaltime;
modele.history_loglik = histLL;
modele.initmodel = initmodel;
%modele.y = yold;
modele.x = xold;
modele.nb_clusters = K;
modele.T = T;
modele.L = L;
modele.conditionsConvergence = [cond5 cond2 cond3 cond5c cond2b];
modele.initmodel = initmodel;
stored_norm(iter:end)=[];
modele.normParameters = stored_norm; 
% compute Criteria
% compute Criteria
p = (K-1)*3 + numel(modele.mu);
if initmodel.sharedCovariances
    p = p + initmodel.dx*(initmodel.dx+1)/2; 
else
    p = p + K*initmodel.dx*(initmodel.dx+1)/2; 
end

%figure, plot(modele.normParameters),title('Norm of xold-xnew through iterations')
%figure, plot(modele.history_loglik),title('Loglikelihood through iterations')

modele.nparams = p; % nb parameters in the model
modele.criteria.AIC = -2*modele.loglik + 2*p; % to be minimised
modele.criteria.BIC = -2*modele.loglik + p*log(modele.L); % to be minimised
% Entropy can be regarded as a penalty for the observed-datalikelihood  
% when the estimated clusters are not well separated
modele.criteria.conditional_entropy = -sum( yold .* log( yold + double(yold == 0)), 'all' );
[valueshard_partition, idxhard_partition] = max(yold,[],2); % max a posteriori (MAP)
c = zeros(K,L); % make 0-1 encoding of MAP 
c(idxhard_partition' + [0:K:K*L-1]) = 1;
modele.criteria.hardclustering_entropy = -sum(sum( c' .* log( yold + double(yold == 0) ), 2),1);
% modele.criteria.softclustering_entropy = -sum(sum( valueshard_partition .* log( yold + double(yold == 0) ), 2), 1);
modele.criteria.ICL = modele.criteria.BIC + 2*modele.criteria.hardclustering_entropy; % mixmod.org, equation 27
% modele.criteria.ICL2 = modele.criteria.BIC + 2*modele.criteria.softclustering_entropy; 
        
% % fprintf('Compute NEC...\n') % mixmod.org page 14 eq 32
% % % With one component, alpha = 1, so only the gaussian component remains
% % mix = gmm(dx, 1, 'full');
% % options = foptions;
% % options(14) = 500; % nb iterations
% % options(1) = -1; % Switch off all messages 
% % warning  off
% % [~,options] = gmmem(mix, data, options); % estime 1 component GMM
% % warning  on
% % L1 = options(8); 
% % modele.criteria.NEC = modele.criteria.conditional_entropy / (modele.loglik - L1);
% 
% % figure('Name','Evolution du critere'),subplot(211),plot(histLL),title('Criterion')
% % subplot(212),plot(evol_nucoeff),title('Evolution du param d''adaptation dans le gradient conj. \nu_{i}')
% % figure('Name','pi_tk'),plot(t,pi),title('\pi_{tk}')
% % figure('Name','y'),plot(t,y),title('y_{tk}')
% % figure('Name','alpha'),plot(t,alpha),,title('\alpha_{tk}')


