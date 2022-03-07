% Plot histograms of onsets for real data. 
% It makes use of results modelesGMMseq.mat generated by 
% test_selection_models_GMMSEQ_via_histTau.m 
% using a specific data structure. Those results were stored into a 
% hard written folder "ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION" where
% modelesGMMseq.mat files were recorded. To be adapted for your case.
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

addpath '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/dev/PROJETS/GMM_TIME/ESSAIS_GMMSEQ_TRUSTR_TAUPRIOR'
addpath '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/dev/PROJETS/GMM_TIME/ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION'


% file names
if 1 % trust region
    s={ 'GMMSEQ_desserrage_GMMseq_B.mat',...
        'GMMSEQ_desserrage_GMMseq_C.mat',...
        'GMMSEQ_desserrage_GMMseq_D.mat',...
        'GMMSEQ_desserrage_GMMseq_E.mat',...
        'GMMSEQ_desserrage_GMMseq_F.mat'}
    noms = {'GMMSEQ applied on \#B','GMMSEQ applied on \#C',...
        'GMMSEQ applied on \#D','GMMSEQ applied on \#E','GMMSEQ applied on \#F'}
    jeux = {'B','C','D','E','F'};
    withpriorstau = {...
        'GMMSEQ_B_trustregion_tauprior_lambda1000',...
        'GMMSEQ_C_trustregion_tauprior_lambda1000',...
        'GMMSEQ_D_trustregion_tauprior_lambda1000',...
        'GMMSEQ_E_trustregion_tauprior_lambda1000',...
        'GMMSEQ_F_trustregion_tauprior_lambda1000'};
    features = {'mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
        'mesure_C_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
        'mesure_D_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
        'mesure_E_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
        'mesure_F_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'}
    
elseif 1 % quasi newton with different nb of iterations
    %s={ 'GMMSEQ_B_quasinewton_1iter',...
    %    'GMMSEQ_C_quasinewton_1iter',...
    %    'GMMSEQ_D_quasinewton_1iter',...
    %    'GMMSEQ_E_quasinewton_1iter',...
    %    'GMMSEQ_F_quasinewton_1iter'}
    s={ 'GMMSEQ_B_quasinewton_1500iter',...
        'GMMSEQ_C_quasinewton_1500iter',...
        'GMMSEQ_D_quasinewton_1500iter',...
        'GMMSEQ_E_quasinewton_1500iter',...
        'GMMSEQ_F_quasinewton_1500iter'}
   noms = {'GMMSEQ applied on \#B','GMMSEQ applied on \#C',...
        'GMMSEQ applied on \#D','GMMSEQ applied on \#E','GMMSEQ applied on \#F'}
    jeux = {'B','C','D','E','F'};
elseif 0 % interactions between features do not bring anything
    s={ 'GMMSEQ_desserrage_B_interactions.mat',...
        'GMMSEQ_desserrage_C_interactions.mat',...
        'GMMSEQ_desserrage_D_interactions.mat',...
        'GMMSEQ_desserrage_E_interactions.mat',...
        'GMMSEQ_desserrage_F_interactions.mat'}
    noms = {'GMMSEQ applied on \#B','GMMSEQ applied on \#C',...
        'GMMSEQ applied on \#D','GMMSEQ applied on \#E','GMMSEQ applied on \#F'}
    jeux = {'B','C','D','E','F'};
end
PASHIST = 2;
 
for ii=1:length(s)
    
    clear modelesGMMseq
    load(['ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION' filesep s{ii}],'modelesGMMseq','lesduree');
    %load(s{ii},'modelesGMMseq','lesduree')
    
    lesduree(lesduree==0) = [];
    lesduree=[0 cumsum(lesduree)]; % verite terrain sur desserrage
    lesduree(end) = [];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % read the structure
    les_tau = []; % all clusters considered
    les_tau7 = []; % 7 clusters only to see variations
    [nessais,nK,nM] = size(modelesGMMseq);
    lesbeta = []; lesgamma = [];
    clear LL, idx = []; k = 1; lesK = [];
    ari = [];
    for ne=1:nessais % trials
        for K=1:nK % clusters
            for m=1:nM % models for init
                if ~isempty(modelesGMMseq{ne,K,m})
                    if modelesGMMseq{ne,K,m}.nb_clusters == 7 % special case - ground truth
                        les_tau7 = [les_tau7, modelesGMMseq{ne,K,m}.tau];
                        lesbeta = [lesbeta, modelesGMMseq{ne,K,m}.beta];
                        lesgamma = [lesgamma, modelesGMMseq{ne,K,m}.gamma];
                    end
                    ari = [ari;modelesGMMseq{ne,K,m}.estim_ARI];
                    les_tau = [les_tau, modelesGMMseq{ne,K,m}.tau];
                    LL(k) =  modelesGMMseq{ne,K,m}.loglik;
                    idx = [idx ; [ne,K,m]];
                    lesK = [lesK; modelesGMMseq{ne,K,m}.nb_clusters];
                    k = k+1;
                end
            end
        end
    end
    %figure,plot(lesbeta),title('Beta')
    %figure,plot(lesgamma),title('Gamma')
    %figure,semilogy(les_tau7,lesbeta,'.'),title('Tau/Beta')
    %figure,semilogy(les_tau7,lesgamma,'.'),title('Tau/Gamma')
    %[p,pi] = GMMSEQ_test(modelesGMMseq{ne,K,m},X,temps);

    %%cc = '/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram/features_articles';
    %%[Xtrain,Ytrain,temps,listFeatures,lesduree] = load_data_irt(cc, features{ii}, false);
      
        
    if 0
        % Case 1: All clusters and all trials and all initialisations
        [a,b]=hist(les_tau,[0:PASHIST:modelesGMMseq{1}.T]);
        a = a/sum(a);
        ffig=figure; hold on, set(gca,'fontsize',22)
        for i=1:length(lesduree)
            d = abs(b-lesduree(i)); [d,e] = min(d);
            line(lesduree(i)*ones(1,10),linspace(a(e),max(a)+0.10*max(a),10),'LineStyle','--','Color',[0 0 0])
        end
        bar(b,a)
        xlabel('Onsets time ($\tau$)','interpreter','latex','fontsize',26),
        ylabel('Normalised histogram of $\tau$ values','interpreter','latex','fontsize',26)
        xlim([-1,65])
        title([noms{ii} ' considering all nb of clusters'],'interpreter','latex','fontsize',26)
        set(ffig,'Position',[1           1        1920         954])
        drawnow
        saveas(ffig,[s{ii}(1:end-4) '_considering_ALL_clusters'],'fig')
        saveas(ffig,[s{ii}(1:end-4) '_considering_ALL_clusters'],'epsc')
    end
    
    if 0
        % Case 2: 7 clusters only
        [a,b]=hist(les_tau7,[0:PASHIST:modelesGMMseq{1}.T]); 
        a = a/sum(a);
        ffig2=figure; hold on, set(gca,'fontsize',22)
        for i=1:length(lesduree)
            d = abs(b-lesduree(i)); [d,e] = min(d);
            line(lesduree(i)*ones(1,10),linspace(a(e),max(a)+0.15*max(a),10),'LineStyle','--','Color',[0 0 0])
        end
        bar(b,a)
        xlabel('Onsets time ($\tau$)','interpreter','latex','fontsize',26),
        ylabel('Normalised histogram of $\tau$ values','interpreter','latex','fontsize',26)
        xlim([-1,65])
        title([noms{ii} ' considering cases with 7 clusters'],'interpreter','latex','fontsize',26)
        set(gcf,'Position',[1           1        1920         954])
        saveas(gcf,[s{ii}(1:end-4) '_considering_7_clusters'],'fig')
        saveas(gcf,[s{ii}(1:end-4) '_considering_7_clusters'],'epsc')
    end  
    
    % Case 3: considering best model in terms of likelihood
    % for each k, select the best model
    u = unique(lesK); u(isinf(u)) = [];
    p = zeros(length(u),1);
    clear bestModel
    ikp = 1;
    for i=1:length(u)
        f = find(lesK==u(i));
        [ll,r] = max(LL(f));
        p(i) = f(r); 
        bestModel{i} = modelesGMMseq{idx(p(i),1),idx(p(i),2),idx(p(i),3)};
        if (contains(lower(s{ii}),'gmmseq_C') && bestModel{i}.nb_clusters == 6) || ...
            (~contains(lower(s{ii}),'gmmseq_C') && bestModel{i}.nb_clusters == 7)
                ikp = i;
                arigmmseqsansprior = bestModel{i}.estim_ARI;               
        end        
    end        
    [[0 10 20 30 40 50 60]; bestModel{ikp}.tau] % should be close
    
    % Plot MDS
    %MDS;
    %
    
    % histogramme des tau sur les meilleurs modeles
    les_tau = [];
    for i=1:length(bestModel)
        les_tau = [les_tau bestModel{i}.tau];
    end
    [a,b]=hist(les_tau,[0:PASHIST:bestModel{1}.T]);
    a = a/sum(a);
    ffig3=figure; hold on, set(gca,'fontsize',22)
    for i=1:length(lesduree)
        d = abs(b-lesduree(i)); [d,e] = min(d);
        line(lesduree(i)*ones(1,10),linspace(a(e),max(a)+0.10*max(a),10),'LineStyle','--','Color',[0 0 0])
    end
    bar(b,a)
    xlabel('Onsets time ($\tau$)','interpreter','latex','fontsize',26),
    ylabel('Normalised histogram of $\tau$ values','interpreter','latex','fontsize',26)
    xlim([-1,65])
    title([noms{ii} ' considering only best (likeliest) clusterings'],'interpreter','latex','fontsize',26)
    set(ffig3,'Position',[1           1        1920         954])
    drawnow
    saveas(ffig3,[s{ii}(1:end-4) '_considering_best_clusterings'],'fig')
    saveas(ffig3,[s{ii}(1:end-4) '_considering_best_clusterings'],'epsc')
    
    les_tau_sans_prior = les_tau;
    
    %################################################################
    
    % superimpose the results with priors on TAU
    
    if exist('withpriorstau')
        
        clear modelesGMMseq
        reswithPriors = load(['ESSAIS_GMMSEQ_TRUSTR_TAUPRIOR' filesep withpriorstau{ii}],'modelesGMMseq');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % read the structure
        les_tau = []; % all clusters considered
        [nessais,nK,nM] = size(reswithPriors.modelesGMMseq);
        clear LL, idx = []; k = 1; lesK = [];
        for ne=1:nessais % trials
            for K=1:nK % clusters
                for m=1:nM % models for init
                    if ~isempty(reswithPriors.modelesGMMseq{ne,K,m})
                        les_tau = [les_tau, reswithPriors.modelesGMMseq{ne,K,m}.tau];
                        LL(k) =  reswithPriors.modelesGMMseq{ne,K,m}.loglik;
                        idx = [idx ; [ne,K,m]];
                        lesK = [lesK; reswithPriors.modelesGMMseq{ne,K,m}.nb_clusters];
                        k = k+1;
                    end
                end
            end
        end
        
        % Case 3: considering best model in terms of likelihood
        % for each k, select the best model
        % with prior only K=7 was treated
        u = unique(lesK); u(isinf(u)) = [];
        p = zeros(length(u),1);
        clear bestModel
        ikp = 1;
        for i=1:length(u)
            f = find(lesK==u(i));
            [ll,r] = max(LL(f));
            p(i) = f(r);
            bestModel{i} = reswithPriors.modelesGMMseq{idx(p(i),1),idx(p(i),2),idx(p(i),3)};
            if (contains(lower(s{ii}),'gmmseq_c') && bestModel{i}.nb_clusters == 6) || ...
                    (~contains(lower(s{ii}),'gmmseq_c') && bestModel{i}.nb_clusters == 7)
                ikp = i;
                arigmmseqavecprior = bestModel{i}.estim_ARI;
            end
        end
        bestModel{1}.estim_ARI, arigmmseqavecprior
        

        % histogramme des tau sur les meilleurs modeles
        les_tau = [];
        for i=1:length(bestModel)
            les_tau = [les_tau bestModel{i}.tau];
        end
        [reswithPriors_a,reswithPriors_b]=hist(les_tau,[0:PASHIST:bestModel{1}.T]);
        reswithPriors_a = reswithPriors_a/sum(reswithPriors_a);
        
        ffig3=figure; hold on, set(gca,'fontsize',22)
        for i=1:length(lesduree)
            d = (abs([b,reswithPriors_b]-lesduree(i))); [d,e] = min(d);
            x = [a,reswithPriors_a]; 
            bb=line(lesduree(i)*ones(1,10),linspace(x(e),max(x)+0.10*max(x),10),'LineStyle','--','Color',[0 0 0]);
        end
        hh=bar([b(:),reswithPriors_b(:)],[a(:),reswithPriors_a(:)],1.1);
        xlabel('Onsets time ($\tau$)','interpreter','latex','fontsize',26),
        ylabel('Normalised histogram of $\tau$ values','interpreter','latex','fontsize',26)
        xlim([-1,65])
        %title([noms{ii} ' considering only best (likeliest) clusterings'],'interpreter','latex','fontsize',26)
        title(noms{ii},'interpreter','latex','fontsize',26)
        set(ffig3,'Position',[1           1        1920         954])
        drawnow
        legend1 = legend([hh,bb],['Considering only best (likeliest) clusterings'],...
            sprintf('With prior on $\\tau_k$ and K=%d (ARI=%2.2f)',bestModel{1}.nb_clusters,bestModel{1}.estim_ARI),...
            'Ground truth','interpreter','latex')
        set(legend1,...
            'Position',[0.65367654907755 0.883643066650207 0.25162648122548 0.0899577599014878],...
            'Interpreter','latex','fontsize',26);
        saveas(ffig3,[s{ii}(1:end-4) '_considering_best_clusterings_with_regul'],'fig')
        saveas(ffig3,[s{ii}(1:end-4) '_considering_best_clusterings_with_regul'],'epsc')
                
        %%%%%%%%%%%%
        % https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall?hl=fr
        % What proportion of positive detection was actually correct?
        % precision: acc=TP/(TP+FP);
        % => A model that produces no false positives has a precision of 1.0.
        % What proportion of actual positive detection was identified correctly?
        % recall: TP/(TP+FN);
        % => A model that produces no false negatives has a recall of 1.0.
        % To compute TP, just take the following rule:
        % if the i-th onset estimated is within [true-t1,true+t2]
        % then TP = TP + 1. If several onsets fall within, only one is counted.
        % To compute FP: it relates to the nb of onsets outside the interval.
        % To compute FN: it relates to the nb of times there is no
        % fall within intervals.
        cc = '/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram/features_articles';
        [Xtrain,Ytrain,temps,listFeatures,lesduree] = load_data_irt(cc, features{ii}, false);
        [onsets_true,pb] = findOnsets(Ytrain,temps,1);
        if pb~=0 && ~contains(lower(s{ii}),'gmmseq_c')
            assert(pb==0);
        elseif contains(lower(s{ii}),'gmmseq_c')
            if numel(find(Ytrain==5))>0
                error('?')
            end
            Ytrain(find(Ytrain==6))=5;
            Ytrain(find(Ytrain==7))=6;
            [onsets_true,pb] = findOnsets(Ytrain,temps,1);
        end
        
        onsets = {les_tau_sans_prior, les_tau};
        t1=0.5; % seconds
        t2=0.5; % seconds
        
        TP = zeros(length(onsets),1);
        FN = zeros(length(onsets),1);
        FP = zeros(length(onsets),1);
        acc = zeros(length(onsets),1);
        rec = zeros(length(onsets),1);
        f1 = zeros(length(onsets),1);
        e = zeros(length(onsets),1);
        
        for j=1:length(onsets)
            
            eval_onsets = zeros(1,length(onsets_true));
            for i = 1:length(onsets{j})
                % is the onsets_est(i) with [onsets_true-t1, onsets_true+t2] ?
                f = find(onsets{j}(i) >= onsets_true-t1 & onsets{j}(i) <= onsets_true+t2);
                if length(f)>1 
                    error('??')
                elseif length(f)==0
                    % Wrong detection is a FP it is outside all intervals
                    FP(j) = FP(j) + 1;
                else
                    eval_onsets(f) = eval_onsets(f)+1;
                end
            end
            FN(j) = sum(eval_onsets == 0);
            TP(j) = sum(eval_onsets > 0); % count only one onsets as positive
            acc(j) = TP(j) / (TP(j) + FP(j));
            rec(j) = TP(j) / (TP(j) + FN(j));
            f1(j) = 2*acc(j)*rec(j) / (acc(j) + rec(j));
            p=eval_onsets/sum(eval_onsets);
            e(j) = -sum(p.*log2(p+(p==0)))/log2(length(p));
        end 
         
        t=array2table([acc,rec,e,[arigmmseqsansprior ; arigmmseqavecprior]],'VariableNames',{'accuracy' 'recall' 'entropy' 'ARI'});
        t=addvars(t, {'GMMSEQ w/o prior on onsets';'GMMSEQ with prior on onsets'}, 'NewVariableNames', 'Algo', 'Before', 'accuracy')
        table2latex(t,['tab_withoroutprior' s{ii}])
    
    end
    
    
    if 0
        % Trace les distances maps pour le meilleur modele pour les moyennes
        % Pairwise distances between means of best model with higher #clusters
        listK = zeros(1,length(bestModel));
        for i=1:length(bestModel), listK(i) = bestModel{i}.nb_clusters; end
        [a,b]=max(listK);
        M = squareform(pdist(bestModel{b}.mu,'mahalanobis',squeeze(mean(bestModel{b}.sigma,1))));
        %M = squareform(pdist(bestModel{b}.tau'));
        
        n = sum(M,'all','omitnan');
        n = n + double(n==0);
        M = M/n;
        
        figure,
        subplot(121),imagesc(M)
        xlabel('Cluster index','interpreter','latex','fontsize',26), ylabel('Cluster index','interpreter','latex','fontsize',26)
        set(gca,'Fontsize',24)
        title(['\#' jeux{ii} ': Pairwise dists of means (likeliest model)'],'interpreter','latex','fontsize',26)
        set(gcf,'Position',[1           1        1920         954])
        colorbar
        %saveas(gcf,['Pairwise_distance_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'fig')
        %saveas(gcf,['Pairwise_distance_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'epsc')
        %saveas(gcf,['Pairwise_distance_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'png')
        
        linkageType = 'ward'; %single complete median average ward
        Z = linkage(M,linkageType);
        subplot(122), dendrogram(Z)
        xlabel('Cluster index','interpreter','latex','fontsize',26), ylabel('Distances','interpreter','latex','fontsize',26)
        set(gca,'Fontsize',24)
        title('Dendogram','interpreter','latex')
        set(gcf,'Position',[1           1        1920         954])
        %saveas(gcf,['Pairwise_distance_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'fig')
        %saveas(gcf,['Pairwise_distance_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'epsc')
        saveas(gcf,['Pairwise_distance_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'png')
    end
    
    if 0 % prefered to use "comparison.m"
        % AIC BIC ICL
        figure,hold on; for i=1:length(bestModel),plot(bestModel(i).nb_clusters,bestModel(i).criteria.AIC,'bo'), end, ylabel('AIC')
        figure,hold on; for i=1:length(bestModel),plot(bestModel(i).nb_clusters,bestModel(i).criteria.BIC,'bo'), end, ylabel('BIC')
        figure,hold on; for i=1:length(bestModel),plot(bestModel(i).nb_clusters,bestModel(i).criteria.ICL,'bo'), end, ylabel('ICL')
        figure,hold on; for i=1:length(bestModel),plot(bestModel(i).nb_clusters,bestModel(i).loglik,'bo'), end, ylabel('LL')
        %now plot de tau values
        
        figure,hold on;
        for i=1:length(bestModel),
            d = diff([sort(bestModel(i).tau) bestModel(i).T]);
            d = sum(d .^2);
            plot(bestModel(i).nb_clusters,d,'bo')
        end
        ylabel('Somme des écarts au carré')
        
        figure,hold on;
        for i=1:length(bestModel),
            d = diff([sort(bestModel(i).tau) bestModel(i).T]);
            d = d/sum(d);
            d = -sum(d .* log2(d));
            plot(bestModel(i).nb_clusters,d,'bo')
        end
        ylabel('Entropie des ecarts')
        
    end
    
end
%[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(les_tau,1,25,0,1e-4,0)
%[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(les_tau+1*randn(1,length(les_tau)),1,14,0,1e-4,0)

return

% Check whether for 14 clusters the means can be "fused" in a coherent way
% wrt the position of the untightening
for ii=1:length(s)
    
    clear modelesGMMseq
    load(['ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION' filesep s{ii}],'modelesGMMseq','lesduree');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    les_means = [];
    les_tau = [];
    [nessais,nK,nM] = size(modelesGMMseq);
    clear LL, idx = []; k = 1; lesK = [];
    for ne=1:nessais
        for K=1:nK
            for m=1:nM
                if ~isempty(modelesGMMseq{ne,K,m})
                    if modelesGMMseq{ne,K,m}.nb_clusters == 14
                        les_means(:,:,end+1) = modelesGMMseq{ne,K,m}.mu;
                        les_tau = [les_tau; modelesGMMseq{ne,K,m}.tau(:)'];
                    end
                    LL(k) =  modelesGMMseq{ne,K,m}.loglik;
                    idx = [idx ; [ne,K,m]];
                    lesK = [lesK; modelesGMMseq{ne,K,m}.nb_clusters];
                    k = k+1;
                end
            end
        end
    end
    
    % Each line in les_means is a set of means for one clustering
    % If we merge the "tau" (onsets), is the fusion coherent for the means?
    %figure(1), clf, figure(2), clf
    for k = 1:size(les_tau,1) % index of the clustering result
        
        S = squareform(pdist(les_tau(k,:)'));
        n = sum(S,'all','omitnan');
        n = n + double(n==0);
        S = S/n;
        if k==1, meanS = S; else meanS = meanS + S; end
        %figure,imagesc(S),title(['PDIST tau values for ' num2str(k)])
        
        M = squareform(pdist(les_means(:,:,k)));
        n = sum(M,'all','omitnan');
        n = n + double(n==0);
        M = M/n;
        if k==1, meanM = M; else meanM = meanM + M; end
        %figure,imagesc(M),title(['PDIST mean values for ' num2str(k)])
        
    end
    meanM = meanM/size(les_tau,1);
    meanS = meanS/size(les_tau,1);
    
    linkageType = 'average'; %single complete median average ward
    Z = linkage(meanM,linkageType);
    figure,   dendrogram(Z)
    set(gca,'Fontsize',24)
    title([jeux{ii} ': Dendogram of means values for ' num2str(size(les_tau,1)) ' clustering results and ' num2str(size(les_tau,2)) ' clusters'],'interpreter','latex','fontsize',26)
    set(gcf,'Position',[1           1        1920         954])
    saveas(gcf,['Dendogram_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'fig')
    saveas(gcf,['Dendogram_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'epsc')
    saveas(gcf,['Dendogram_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'png')
    
    Z = linkage(meanS,linkageType);
    figure,   dendrogram(Z)
    set(gca,'Fontsize',24)
    title([jeux{ii} ': Dendogram of tau values for ' num2str(size(les_tau,1)) ' clustering results and ' num2str(size(les_tau,2)) ' clusters'],'interpreter','latex','fontsize',26)
    set(gcf,'Position',[1           1        1920         954])
    saveas(gcf,['Dendogram_tau_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'fig')
    saveas(gcf,['Dendogram_tau_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'epsc')
    saveas(gcf,['Dendogram_tau_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'png')
    
    
    figure,imagesc(meanS)
    colorbar
    title([jeux{ii} ': Average pairwise distance of tau values for ' num2str(size(les_tau,1)) ' clustering results and ' num2str(size(les_tau,2)) ' clusters'],'interpreter','latex','fontsize',26)
    set(gca,'Fontsize',24)
    set(gcf,'Position',[1           1        1920         954])
    xlabel('Cluster index','interpreter','latex','fontsize',26), ylabel('Cluster index','interpreter','latex','fontsize',26)
    saveas(gcf,['Average_pdist_tau_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'fig')
    saveas(gcf,['Average_pdist_tau_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'epsc')
    saveas(gcf,['Average_pdist_tau_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'png')
    
    figure,imagesc(meanM)
    colorbar
    set(gca,'Fontsize',24)
    title([jeux{ii} ': Average pairwise distance of means values for ' num2str(k) ' clustering results and ' num2str(size(les_tau,2)) ' clusters'],'interpreter','latex','fontsize',26)
    set(gcf,'Position',[1           1        1920         954])
    xlabel('Cluster index','interpreter','latex','fontsize',26), ylabel('Cluster index','interpreter','latex','fontsize',26)
    saveas(gcf,['Average_pdist_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'fig')
    saveas(gcf,['Average_pdist_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'epsc')
    saveas(gcf,['Average_pdist_means_' num2str(size(les_tau,1)) 'clust_results_' s{ii}(1:end-4)],'png')
    
end


