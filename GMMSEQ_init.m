function initmodel = GMMSEQ_init(X,temps,K,meth,optimMethod,useMiniBatches,sharedCovariances)
% initmodel is the initial guess of parameters
% X = data
% temps = time vector
% K = nb clusters
% meth = initialisation method "gmm" or "kmeans" or "segments"
% optimMethod = "matlab-explicitGrad" => uses Matlab Opt. toolbox for
% optimising parameters with explicit gradients + trust region method
% instead of linesearch (best but slow)
% optimMethod = "matlab-quasiNewton" => use Matlab Opt. toolbox for 
% optimising parameters with explicit gradients (2020b version) with 
% Wolfe linesearch + numerical estimate of Hessian (very good)
% optimMethod = "schmidt" => use" Schmidt toolbox for optimising parameters 
% with explicit gradients (quite good)
% optimMethod = "linesearch" => use an external function to make a
% linesearch (not good... for the tests made)
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% init model of clusters with GMM or KMEANS
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dim = size(X,2);
initmodel.dx = dim;               % dimension of data
initmodel.meth = meth;            % method MVT or GMM
initmodel.nb_clusters = K;        % nb of clusters
initmodel.T = max(temps);         % max time
initmodel.L = size(X,1);          % size of data
initmodel.seuilMinNu = 1e-6;      % method MVT only: minimal degrees of freedom, if > 1 means are defined, > 2 variances are defined
initmodel.df = 0.5*ones(1,K);     % method MVT only: initial degrees of freedom
initmodel.amountProbMin = eps;    % add some minimal prob to u and y
initmodel.nitermax = 5000;       % to stop iterations during learning
initmodel.plotDuringIter = false; % Display results during iterations
initmodel.minvar = 1e-6;          % minimum variance for initialisation of covariances matrices
initmodel.thnorm2 = 1e-2;         % threshold for ||xnew-xold|| = norm(xnew - xold); 
initmodel.thresholdLLdiff = 1e-3; % threshold on the evolution of the likelihood
initmodel.useMiniBatches = useMiniBatches; % either use whole dataset or sample from it to evaluate gradients
initmodel.sizeMiniBatches = 2048; % size of batch for useMiniBatches = true
initmodel.verbosity = false;      % set verbosity of the algorithm during iteration
initmodel.sharedCovariances = sharedCovariances; % share or not covariances between clusters

% In case the statistical toolbox is not present use external (maybe less efficient) programs
try                              
    mvnpdf(0,1,1); % if this passed then ok
    initmodel.useExternalFunctions = false;
catch 
    initmodel.useExternalFunctions = true;
end

% Parameters for the optimisation algorithm
initmodel.penalisation = [];
initmodel.UseParallel = true; % use parallel if gradients are not used
switch lower(optimMethod)
    % MATLAB toolbox with trust region algorithm
    % Set here the number of iterations 1 per default is fast but not reliable
    % works very well for high iterations but is slow for high dimensions
    case "matlab-explicitgrad" 
        initmodel.minFunc.name = optimMethod;
        initmodel.UseExplicitGradient = true;
        options = optimoptions(...
            'fminunc',...
            'Algorithm','trust-region',...
            'SpecifyObjectiveGradient',initmodel.UseExplicitGradient,...
            'Display','off', ... %'CheckGradients',true, 'FiniteDifferenceType', 'central', ...
            'UseParallel',initmodel.UseParallel);
        options.MaxIterations = 1500;
        initmodel.minFunc.options = options;
        
        if useMiniBatches==false && size(X,1) > 2000 && options.MaxIterations > 10
            warning('You should set useMiniBatches=true with matlab-explicitgrad + size(X,1)>2000 with high nb of iterations')
        end
        
    % MATLAB toolbox with quasi Newton method
    % when using explicit gradients, linesearch with Wolfe method is done
    % Set here the number of iterations 1 per default 
    % Some errors sometimes - just rerun the algorithm, it will converge!
    case "matlab-quasinewton"
        initmodel.UseExplicitGradient = true; % if false uses parallel toolbox and the numerical estimation of gradients
        initmodel.minFunc.name = optimMethod;
        options = optimoptions(...
            'fminunc',...
            'Algorithm','quasi-newton',...
            'SpecifyObjectiveGradient',initmodel.UseExplicitGradient,...
            'Display','off', ... %'CheckGradients',true, 'FiniteDifferenceType', 'central', ...
            'UseParallel',initmodel.UseParallel);
        options.MaxIterations = 1500;
        initmodel.minFunc.options = options;
        
    % SCHMIDT Toolbox MinFUNC using L-BFGS + Wolfe linesearch       
    % Alternative to Matlab, but not as good results as two previous ones
    % with GMMSEQ
    case "schmidt" 
        initmodel.minFunc.name = optimMethod;
        clear options
        options.Method = 'lbfgs';  %sd | csd | bb | cg | scg | pcg | {lbfgs} | newton0 | pnewton0 | qnewton | mnewton | newton | tensor
        options.display = 'off';
        %options.MaxIter = 1;
        initmodel.minFunc.options = options;
        
    % simple implementation of linesearch
    % Not reliable with GMMSEQ
    case "linesearch"
        initmodel.minFunc.name = optimMethod;
        % clear options
        % options.mode       = 'polak'; % 'fletcher', 'hestenes' or 'polak'
        % options.linesearch = 'exact'; % 'armijo', 'quadratic' or 'exact'
        % options.verbosity = false;
        % options.numIterations = 1;
        % options.maxStepLength = 5;
        % initmodel.minFunc.options = options;
        
    otherwise
        error('Unknown optimMethod: matlab-explicitgrad or matlab-quasinewton or linesearch');
end

% Three different initialisations: Kmeans, GMM or segments
% For each of them, look for the onset estimated
if strcmp(meth,'kmeans')
    
    disp('Initialise clusters with Kmeans...')
     
    % init with Kmeans    
    initmodel.covType = 'full';
    initmodel.sigma = zeros(K,dim,dim); 
    
    if not(initmodel.useExternalFunctions)
        % Matlab version - look for best kmeans over 10 trials
        %[a,c] = kmeans(X,K,'replicates',10,'EmptyAction','error');
        [a,c] = kmeans(X,K,'EmptyAction','error');
    else
        % Netlab
        [c, ~, a, ~] = kmeans(K, X);
    end
    
    initmodel.initialPartition = a;
    initmodel.mu = c; 
    initmodel.mixingCoeff = zeros(1,K);
    for i=1:K
        aa = a==i;
        initmodel.sigma(i,:,:) = cov(X(aa,:));
        initmodel.mixingCoeff(i) = sum(aa);
    end
    initmodel.mixingCoeff = initmodel.mixingCoeff/size(X,1); 
    if initmodel.sharedCovariances
        initmodel.covType = 'SharedCovariance';
        m = squeeze(mean(initmodel.sigma,1));
        for k=1:K
            initmodel.sigma(k,:,:) = m;
        end
    end
    
elseif strcmp(meth,'gmm')
    
    % here we can optimise the gmm components
    % or set iter = 1 to optimise it in the next lines of code
    % iter = 1000 will make the next faster but will compromise the chance to
    % find other patterns when components are coupled with sigmoids
    % ...similar to continuation learning Goodfellow p 318
    
    disp('Initialise clusters with GMM...')
    
    if not(initmodel.useExternalFunctions)
        
        try
            options=statset;
            options.MaxIter=1000; 
            if initmodel.sharedCovariances
                gmfit= fitgmdist(X,K,...
                    'RegularizationValue',0.01,...
                    'CovarianceType','full',...
                    'SharedCovariance',true,...  %  'Replicates',10,...
                    'options',options); %diagonal
                initmodel.covType = 'SharedCovariance';
            else
                gmfit= fitgmdist(X,K,...
                    'RegularizationValue',0.01,...
                    'CovarianceType','full',... % 'Replicates',10,...
                    'options',options); %diagonal
            end
            
            initmodel.gmmmodel = gmfit;% all characteristics of clusters init are stored
            
            % get means and covariance
            for i=1:initmodel.nb_clusters
                if gmfit.SharedCovariance == true
                    initmodel.covType = 'SharedCovariance';
                    initmodel.sigma(i,:,:) = gmfit.Sigma;
                elseif strcmp(gmfit.CovarianceType,'diagonal')
                    initmodel.covType = 'diag';
                    initmodel.sigma(i,:,:) = diag(gmfit.Sigma(1,:,i));
                else
                    initmodel.covType = 'full';
                    initmodel.sigma(i,:,:) = gmfit.Sigma(:,:,i);
                end
                initmodel.mu(i,:) = gmfit.mu(i,:);
            end
            % not used in our model but below for initialising beta
            initmodel.mixingCoeff = gmfit.ComponentProportion;
            
            ok = true;
            
            [~,~,P] = cluster(gmfit,X);
            [~,a] = max(P,[],2);
            for i=1:initmodel.nb_clusters
                f = find(a==i,1,'first');
                assert(~isempty(f));
            end
            
        catch
            % nothing
            ok = false;
        end
    end
    
    if initmodel.useExternalFunctions || ok == false
        
        disp('Use GMM-netlab...')
        % Set up mixture model
        mix = gmm(dim, K, 'full');
        initmodel.covType = 'full';
        options = foptions;
        options(14) = 500; % A single iteration
        options(1) = -1; % Switch off all messages, including warning
        mix = gmmem(mix, X, options);
        for i=1:initmodel.nb_clusters
            initmodel.sigma(i,:,:) = mix.covars(:,:,i);
        end
        initmodel.mu = mix.centres;
        initmodel.mixingCoeff  = mix.priors;
        if initmodel.sharedCovariances
            error('TO BE IMPLEMENTED')
        end
    end

elseif strcmp(meth,'segments')
   
    disp('Initialise clusters with segments...')
    
    dd = round(initmodel.L/K);
    r = 1; 
    a = K*ones(initmodel.L,1); % le K est important!
    
    % fit one Gaussian per segment
    for i=1:K        
        f = r:min(r+dd-1,size(X,1));
        
        if not(initmodel.useExternalFunctions)            
            gmfit= fitgmdist(X(f,:),1,...
                'RegularizationValue',0.01,...
                'CovarianceType','full',...  %  'Replicates',10,
                'MaxIter',500);            
            if strcmp(gmfit.CovarianceType,'diagonal')
                initmodel.covType = 'diag';
                initmodel.sigma(i,:,:) = diag(gmfit.Sigma);
            else
                initmodel.sigma(i,:,:) = gmfit.Sigma;
            end
            initmodel.mu(i,:) = gmfit.mu;

        else
            
            disp('Use GMM-netlab...')
            % Set up mixture model
            mix = gmm(dim, 1, 'full');
            initmodel.covType = 'full';
            options = foptions;
            options(14) = 500; % A single iteration
            options(1) = -1; % Switch off all messages, including warning
            mix = gmmem(mix, X(f,:), options);
            initmodel.sigma(i,:,:) = mix.covars(:,:,1);
            initmodel.mu(i,:) = mix.centres;
            if initmodel.sharedCovariances
                error('TO BE IMPLEMENTED')
            end
            
        end
        a(f) = i;
        r = r + dd;        
    end
    disp('Tabulate of initial clusters:')
    tabulate(a)
    initmodel.initialPartition = a;
    initmodel.mixingCoeff = ones(1,K)/K;
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% init tau, gamma and beta 
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(meth,'gmm')
    
    if not(initmodel.useExternalFunctions)
        
        % first get the clustering
        [~,~,P] = cluster(gmfit,X);
        [~,a] = max(P,[],2);
        disp('Tabulate of initial clusters:')
        tabulate(a)
        initmodel.initialPartition = a;
        
    else
        
        P = gmmpost(mix, X);
        [~,a] = max(P,[],2);
        initmodel.initialPartition = a;
        for i=1:K
            fprintf('Cluster %d \t freq=%d\n',i,length(find(a==i)));
        end
        
    end
end

% then reorder clusters according to order of occurrence
ordreApparition = zeros(2,initmodel.nb_clusters);
for i=1:initmodel.nb_clusters
    ordreApparition(1,i) = i;
    f = find(a==i,1,'first');
    if isempty(f), error('Rerun, an issue occurs during initialisation (empty cluster)'); end
    ordreApparition(2,i) = f;
end
[c, d] = sort(ordreApparition(2,:),'ascend');
ordreApparitionAscend = c;
% disp('Order of occurrence: ')
% disp(ordreApparitionAscend)

% reorder parameters accordingly
muold = initmodel.mu;
sigmaold = initmodel.sigma;
mixingCoeffold = initmodel.mixingCoeff;
for i=1:initmodel.nb_clusters
    initmodel.sigma(i,:,:) = squeeze(sigmaold(d(i),:,:)) + initmodel.minvar*eye(dim);% cov
    initmodel.mu(i,:) = muold(d(i),:);% means
    initmodel.mixingCoeff(i) = mixingCoeffold(d(i));
end

% then initialise beta, gamma and tau

% gamma is the slope
initmodel.gamma = rand(1,initmodel.nb_clusters);

% beta is the amplitude - initialised with proportions weights
initmodel.beta = rand(1,initmodel.nb_clusters);

% tau is the onset time
% disp('Tau regularly spaced in time')
% initmodel.tau = linspace(0,(K-1)/K,K)*(max(temps)-min(temps))+min(temps); %[0 cumsum(ones(1,initmodel.nb_clusters-1)/(initmodel.nb_clusters)) * (max(temps)-min(temps))];
disp('Tau initialised using starting points of clusters') % more relevant!!
initmodel.tau = temps(ordreApparitionAscend)';
% disp('Tau initialised at random')
% initmodel.tau = sort(randi(T,[1,initmodel.nb_clusters]));
fprintf('Tau (Tmax=%f) : \n',max(temps))
disp(initmodel.tau)

