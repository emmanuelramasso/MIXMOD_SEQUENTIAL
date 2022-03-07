% Allows to compare with Kmeans, GMM and hierarchical clustering
% This file is probably the less documented but follows test_selection_models_GMMSEQ_via_histTau.m
%
%   REFERENCE
%   
%   This implementation has been made and shared in the context of a 
%   project between FEMTO-ST (Besançon, France) and UTC (Compiègne, France)
%   leading to the following paper:
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

pathdefmixmod;

dataCampaign = input('Data B C D E F ? : ')
desserrage = true;

%%%%%VERIFIER DOSSIER!% c ='/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram/';
% c = '/home/emmanuel.ramasso/Documents/DATA/IRT/avecHitDetectionEtScalogram';
% c = '/home/emmanuel.ramasso/Documents/CODES/MATLAB';
c = '/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram/features_articles';
% rr = '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/dev/PROJETS/GMM_TIME/GMM_new/VARIANTES_OPTIMIZERS'
% %rr = '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/dev/PROJETS/GMM_TIME/GMM_new1';
% %rr = '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/dev/GMM_test/VARIANTES_OPTIMIZERS';
rr = '/home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/dev/PROJETS/GMM_TIME/ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION';

switch dataCampaign
    case 'B'
        n = 'mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
        s = 'GMMSEQ_desserrage_GMMseq_B.mat';
        %s = 'GMMSEQ_B_quasinewton_1iter'
        %s = 'GMMSEQ_B_quasinewton_1500iter'
        
    case 'C'
        n = 'mesure_C_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
        s = 'GMMSEQ_desserrage_GMMseq_C.mat';
        %s = 'GMMSEQ_C_quasinewton_1iter'
        %s = 'GMMSEQ_C_quasinewton_1500iter'
        
    case 'D'
        n = 'mesure_D_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
        s = 'GMMSEQ_desserrage_GMMseq_D.mat';
        %s = 'GMMSEQ_D_quasinewton_1iter'
        %s = 'GMMSEQ_D_quasinewton_1500iter'
        
    case 'E'
        n = 'mesure_E_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
        s = 'GMMSEQ_desserrage_GMMseq_E.mat';
        %s = 'GMMSEQ_E_quasinewton_1iter'
        %s = 'GMMSEQ_E_quasinewton_1500iter'
        
    case 'F'
        n = 'mesure_F_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
        s = 'GMMSEQ_desserrage_GMMseq_F.mat';
        %s = 'GMMSEQ_F_quasinewton_1iter'
        %s = 'GMMSEQ_F_quasinewton_1500iter'
        
    otherwise error('??')
end

% charge les données
useInteract = false;
[Xtrain,Ytrain,temps,listFeatures,lesduree,nbFeatInit] = load_data_irt(c, n, useInteract);
% Xtrain = zscore(Xtrain(:,1:length(listFeatures))); % SANS INTERACTION
[Xtrain, mD, sD] = zscore(Xtrain); % PREND TOUT

%rng('default')
%Y = tsne(Xtrain(1:10:end,:),'Algorithm','exact','Standardize',true,'Perplexity',20);
% figure; gscatter(Y(:,1),Y(:,2),Ytrain(1:10:end));

lesduree(lesduree==0) = [];
lesduree=[0 cumsum(lesduree)]; % verite terrain sur desserrage
lesduree(end) = [];

% in case pca raises an error it means you do not have the statistical
% toolbox. In this case adapt the code with your own pca code.
[coeff, score, latent, tsquared, explained] = pca(Xtrain);
f = find(cumsum(explained)>99,1,'first');
X = score(:,1:f);
figure,g=gscatter(X(:,1),X(:,2),Ytrain); title('PCA1-2')
set(gca,'fontsize',22)
for i=1:length(g), set(g(i),'Marker','s'), end
xlabel('PC1'),ylabel('PC2')
% plotmatrix_mine(X(:,1:min(10,size(X,2))),Ytrain)

if 0
    rng('default')
    Y = tsne(X(1:10:end,:),...
        'Algorithm','exact',...
        'Standardize',false,...
        'Perplexity',50,...
        'Distance','euclidean',...
        'LearnRate',1000);

    figure; g=gscatter(Y(:,1),Y(:,2),Ytrain(1:10:end));
    % title('Euclid perplexity 10')
    s=['o' 'x' 's' '^' '*' 'd' 'p'];
    for i=1:length(g)
        set(g(i),'Markersize',10)
        set(g(i),'Marker',s(i))
        %set(g(i),'Color',[0 0 1])
    end
    set(gca,'Fontsize',28)
    ylabel('tSNE 1'), xlabel('tSNE 2')
    exportgraphics(gcf, ['TSNE_' dataCampaign 'perp10.pdf'])
    exportgraphics(gcf, ['TSNE_' dataCampaign 'perp10.eps'])

    % add subfolders in
    % /home/emmanuelramasso/OneDrive/Documents/PROGRAMMES/archives_et_autres/AUTRESCODES/tSNE_variabledegreefreedom
    %https://github.com/dkobak/finer-tsne/blob/master/heavy-tailed-tsne.ipynb
    opts.max_iter = 1000;
    opts.df = 1;
    opts.perplexity = 30;
    cluster_firstphase = fast_tsne(X, opts);
    figure,g=gscatter(cluster_firstphase(:,1),cluster_firstphase(:,2),Ytrain);


end

Kall = 4:14 % nb of clusters

if 0
    % CHOIX DES CLUSTERS AVEC CRITERES STANDARDS
    E_kmeans_DB = evalclusters(X,'kmeans','DaviesBouldin','klist',Kall);
    figure,plot(E_kmeans_DB);
    
    E_kmeans_CH = evalclusters(X,'kmeans','CalinskiHarabasz','klist',Kall);
    figure,plot(E_kmeans_CH);
    
    E_gmm_DB = evalclusters(X,'gmdistribution','DaviesBouldin','klist',Kall);
    figure,plot(E_gmm_DB);
    
    E_gmm_CH = evalclusters(X,'gmdistribution','CalinskiHarabasz','klist',Kall);
    figure,plot(E_gmm_CH);
    
    E_link_DB = evalclusters(X,'linkage','DaviesBouldin','klist',Kall);
    figure,plot(E_link_DB);
    
    E_link_CH = evalclusters(X,'linkage','CalinskiHarabasz','klist',Kall);
    figure,plot(E_link_CH);
    
    % FAIRE GMM+AIC et BIC
    % - - - -
end


close all

% CHOIX DES CLUSTERS AVEC CRITERES ONSETS
clear noms onsets 
ari = zeros(4,length(Kall)); % ARI for each number of clusters
[onsets{1},ari(1,:)]=apply_clustering_standard(X,temps,'kmeans','DaviesBouldin',Kall,dataCampaign,lesduree,Ytrain);
noms{1}='Kmeans';
[onsets{2},ari(2,:)]=apply_clustering_standard(X,temps,'gmdistribution','DaviesBouldin',Kall,dataCampaign,lesduree,Ytrain);
noms{2}='GMM';
[onsets{3},ari(3,:)]=apply_clustering_standard(X,temps,'linkage','DaviesBouldin',Kall,dataCampaign,lesduree,Ytrain);
noms{3}='HC';
[onsets{4},ari(4,:)]=apply_clustering_standard(X,temps,'GK','DaviesBouldin',Kall,dataCampaign,lesduree,Ytrain);
noms{4}='GK';


o=cell2mat(onsets); 
leg={};
PASHIST = 3;
[a,b]=hist(o,[0:PASHIST:max(temps)]);
a = a./sum(a,1);
for i=1:size(a,2), leg = [leg, noms{i}]; end

clear modelesGMMseq
%load(['ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION' filesep s],'modelesGMMseq');
load([rr filesep s],'modelesGMMseq');

les_tau = []; % all clusters considered
[nessais,nK,nM] = size(modelesGMMseq);
clear LL, idx = []; k = 1; lesK = [];
for ne=1:nessais
    for K=1:nK
        for m=1:nM
            if ~isempty(modelesGMMseq{ne,K,m}) && ismember(modelesGMMseq{ne,K,m}.nb_clusters,Kall)
                les_tau = [les_tau, modelesGMMseq{ne,K,m}.tau];
                LL(k) =  modelesGMMseq{ne,K,m}.loglik;
                idx = [idx ; [ne,K,m]];
                lesK = [lesK; modelesGMMseq{ne,K,m}.nb_clusters];
                k = k+1;
            end
        end
    end
end

% for each k, select the best model
u = unique(lesK); u(isinf(u)) = [];
p = zeros(length(u),1);
clear bestModel
ikp = 1; arigmmseq = [];
for i=1:length(u)
    f = find(lesK==u(i));
    [ll,r] = max(LL(f));
    p(i) = f(r);
    bestModel{i} = modelesGMMseq{idx(p(i),1),idx(p(i),2),idx(p(i),3)};
    arigmmseq = [arigmmseq; bestModel{i}.estim_ARI];
end

% histogramme des tau sur les meilleurs modeles
les_tau = [];
for i=1:length(bestModel)
    les_tau = [les_tau bestModel{i}.tau];
end
onsets{end+1} = les_tau(:);
noms{end+1} = 'GMMSEQ';

% All clusters
[a1,b1]=hist(les_tau,[0:PASHIST:max(temps)]);
a1 = a1./sum(a1);
a=[a a1(:)];
leg = [leg, 'GMMSEQ'];

figure
c=bar(b,a,'grouped','linestyle','none');%,'BarWidth',0.5);
hold on
for i=1:length(lesduree)
    d = abs(b-lesduree(i)); [d,e] = min(d);
    line(lesduree(i)*ones(1,10),linspace(0,max(a,[],'all')+0.10*max(a,[],'all'),10),'LineStyle','--','Color',[0 0 0])
end
leg = [leg, 'Untightening'];
for i=1:length(c), c(i).BarWidth = 1; end
l=legend(leg);
l.Location = 'eastoutside';
set(gca,'fontsize',24)

xlabel('Onsets time ($\tau$)','interpreter','latex','fontsize',26),
ylabel(['Normalised histogram of onsets for data ' dataCampaign],'interpreter','latex','fontsize',26)
xlim([-1,65])
set(gcf,'Position',[1           1        1920         954])

saveas(gcf,sprintf('data_%s_histo_compare',dataCampaign),'png')
saveas(gcf,sprintf('data_%s_histo_compare',dataCampaign),'epsc')
saveas(gcf,sprintf('data_%s_histo_compare',dataCampaign),'fig')


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
[onsets_true,pb] = findOnsets(Ytrain,temps,1);
if pb~=0 && ~strcmp(dataCampaign,'C')
    assert(pb==0);
elseif strcmp(dataCampaign,'C')
    if numel(find(Ytrain==5))>0
        error('?')
    end
    Ytrain(find(Ytrain==6))=5;
    Ytrain(find(Ytrain==7))=6;
    [onsets_true,pb] = findOnsets(Ytrain,temps,1);
end

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
t=array2table([acc,rec,f1,e,TP,FP,FN],'VariableNames',{'accuracy' 'recall' 'f1' 'entropy' 'TP' 'FP' 'FN'});
t=addvars(t, noms(:), 'NewVariableNames', 'Algo', 'Before', 'accuracy')

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARI PERF
figure,hold on
plot(Kall,ari(1,:),'s-.','linewidth',2,'markersize',10)
plot(Kall,ari(2,:),'x-.','linewidth',2,'markersize',10)
plot(Kall,ari(3,:),'o-.','linewidth',2,'markersize',10)
plot(Kall,ari(4,:),'^-.','linewidth',2,'markersize',10)
h=plot(Kall,arigmmseq,'d-.','linewidth',3,'markersize',10);
set(h,'MarkerFaceColor',h.Color);
legend('Kmeans','GMM','Linkage','GK','GMMSEQ')
set(gca,'fontsize',24)

xlabel('Number of clusters','interpreter','latex','fontsize',26),
ylabel(['ARI (perf) for data ' dataCampaign],'interpreter','latex','fontsize',26)
set(gcf,'Position',[1           1        1920         954])
grid on

saveas(gcf,sprintf('data_%s_Ari_compare',dataCampaign),'png')
saveas(gcf,sprintf('data_%s_Ari_compare',dataCampaign),'epsc')
saveas(gcf,sprintf('data_%s_Ari_compare',dataCampaign),'fig')

if strcmp(dataCampaign,'C')
    f = find(Kall==5 | Kall==6 | Kall==7);
else
    f = find(Kall==6 | Kall==7 | Kall==8);
end
t=array2table([acc,rec,e,mean([ari(:,f); arigmmseq(f)'],2)],'VariableNames',{'accuracy' 'recall' 'entropy' 'ARI'});
t=addvars(t, noms(:), 'NewVariableNames', 'Algo', 'Before', 'accuracy')
table2latex(t,['tab_' dataCampaign])


