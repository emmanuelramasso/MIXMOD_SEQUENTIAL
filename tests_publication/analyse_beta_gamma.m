% Plot some results to show the relevance of the estimated parameters beta
% and gamma in particular
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
% trust region
s={ 'GMMSEQ_desserrage_GMMseq_B.mat',...
    'GMMSEQ_desserrage_GMMseq_C.mat',...
    'GMMSEQ_desserrage_GMMseq_D.mat',...
    'GMMSEQ_desserrage_GMMseq_E.mat',...
    'GMMSEQ_desserrage_GMMseq_F.mat'}
noms = {'GMMSEQ applied on \#B','GMMSEQ applied on \#C',...
    'GMMSEQ applied on \#D','GMMSEQ applied on \#E','GMMSEQ applied on \#F'}
jeux = {'B','C','D','E','F'};
features = {'mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
    'mesure_C_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
    'mesure_D_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
    'mesure_E_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat', ...
    'mesure_F_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'}

PASHIST = 2;

for ii=1:length(s)
    
    clear modelesGMMseq
    load(['ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION' filesep s{ii}],'modelesGMMseq','lesduree');
    
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
                    if 1%~strcmp(jeux{ii},'C') && modelesGMMseq{ne,K,m}.nb_clusters == 7 || strcmp(jeux{ii},'C') && modelesGMMseq{ne,K,m}.nb_clusters == 6 
                        les_tau7 = [les_tau7, modelesGMMseq{ne,K,m}.tau];
                        lesbeta = [lesbeta, modelesGMMseq{ne,K,m}.beta];
                        lesgamma = [lesgamma, modelesGMMseq{ne,K,m}.gamma];
                        ari = [ari;modelesGMMseq{ne,K,m}.estim_ARI];
                        idx = [idx ; [ne,K,m]];
                        lesK = [lesK; modelesGMMseq{ne,K,m}.nb_clusters];
                    end
                    k = k+1;
                end
            end
        end
    end
    
%     figure,plot(lesbeta),title('Beta for 7')
%     figure,plot(lesgamma),title('Gamma for 7')
%     figure,semilogy(les_tau7,lesbeta,'.'),title('Tau/Beta for 7')
%     figure,semilogy(les_tau7,lesgamma,'.'),title('Tau/Gamma for 7')
     
    % load data
    cc = '/home/emmanuelramasso/OneDrive/Documents/RECHERCHE/3-PROJETS/Coalescence_IRT/manip ORION/mars 2019/session 6/featureExtraction/avecHitDetectionEtScalogram/features_articles';
    [Xtrain,Ytrain,temps,listFeatures,lesduree] = load_data_irt(cc, features{ii}, false);
    [Xtrain, mD, sD] = zscore(Xtrain);
    
    lesduree(lesduree==0) = [];
    lesduree=[0 cumsum(lesduree)]; % verite terrain sur desserrage
    lesduree(end) = [];
    
    [coeff, score, latent, tsquared, explained] = pca(Xtrain);
    f = find(cumsum(explained)>99,1,'first');
    X = score(:,1:f);
    
    % select a good model
    [arimax,b] = max(ari);
    [p,pi] = GMMSEQ_test(modelesGMMseq{idx(b(1),1),idx(b(1),2),idx(b(1),3)},X,temps);
    figure1 = figure('WindowState','maximized');
    axes1 = axes('Parent',figure1,...
        'Position',[0.086038961038961 0.11 0.751623376623377 0.815]);
    hold(axes1,'on');
    plot(temps,pi,'linewidth',2),%title('PItk for best ARI')
    set(gca,'Fontsize',26)
    xlabel('Time (s)','interpreter','latex','fontsize',38),
    ylabel('$\pi_{tk}$ (sigmoids responses)','interpreter','latex','fontsize',38)
    xlim([-1,temps(end)])
       
    for i=1:length(lesduree)
        l=line(lesduree(i)*ones(1,10),linspace(0,max(pi(:)),10),'LineStyle','--','Color',[0 0 0])
    end
    l={}; k = modelesGMMseq{idx(b(1),1),idx(b(1),2),idx(b(1),3)}.nb_clusters;
    for i=1:k
        l{end+1} = ['Cluster ' num2str(i)];
    end
    l{end+1} = 'Ground truth';

    legend1 = legend(l)
    set(legend1,...
    'Position',[0.849747477234668 0.447152338130628 0.137445884172147 0.475366862440009]);

    annotation(figure1,'textbox',...
    [0.852190476190476 0.314465408805032 0.132658008658009 0.0911949685534605],...
    'String',{['ARI = ' num2str(round(arimax*1000)/1000)]},...
    'LineStyle','none',...
    'FontSize',24,...
    'FitBoxToText','off');

    saveas(gcf,['pitk_' s{ii}(1:end-4)],'fig')
    exportgraphics(gcf,['pitk_' s{ii}(1:end-4) '.eps'])
    exportgraphics(gcf,['pitk_' s{ii}(1:end-4) '.pdf'])
end



