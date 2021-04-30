% Read AIC et BIC for real data and create plots
% Same comments as in histogrammes_onsets_postClustering_2.m 
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
% Emmanuel Ramasso and Thierry Denoeux
% emmanuel.ramasso@femto-st.fr
% April 2021


clear all
close all

if 1
    s={ 'GMMSEQ_desserrage_GMMseq_B.mat',...
        'GMMSEQ_desserrage_GMMseq_C.mat',...
        'GMMSEQ_desserrage_GMMseq_D.mat',...
        'GMMSEQ_desserrage_GMMseq_E.mat',...
        'GMMSEQ_desserrage_GMMseq_F.mat'}
    noms = {'GMMSEQ applied on \#B','GMMSEQ applied on \#C',...
        'GMMSEQ applied on \#D','GMMSEQ applied on \#E','GMMSEQ applied on \#F'}
    jeux = {'B','C','D','E','F'};
else
    s={ 'GMMSEQ_desserrage_B_interactions.mat',...
        'GMMSEQ_desserrage_C_interactions.mat',...
        'GMMSEQ_desserrage_D_interactions.mat',...
        'GMMSEQ_desserrage_E_interactions.mat',...
        'GMMSEQ_desserrage_F_interactions.mat'}
    noms = {'GMMSEQ applied on \#B','GMMSEQ applied on \#C',...
        'GMMSEQ applied on \#D','GMMSEQ applied on \#E','GMMSEQ applied on \#F'}
    jeux = {'B','C','D','E','F'};
end


for ii=1:length(s)
    
    clear modelesGMMseq
    load(['ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_SANS_INTERACTION' filesep s{ii}],'modelesGMMseq','lesduree');
    %load(['ESSAIS_GMMSEQ_DESSERRAGE_CALCULE_AVEC_INTERACTION' filesep s{ii}],'modelesGMMseq','lesduree');
    
    les_tau = [];
    [nessais,nK,nM] = size(modelesGMMseq);
    C = []; k = 1; clear LL, idx = [];
    for ne=1:nessais
        for K=1:nK
            for m=1:nM
                if ~isempty(modelesGMMseq{ne,K,m})
                    C = [C ; [...
                        modelesGMMseq{ne,K,m}.nb_clusters
                        modelesGMMseq{ne,K,m}.criteria.AIC
                        modelesGMMseq{ne,K,m}.criteria.BIC
                        modelesGMMseq{ne,K,m}.criteria.ICL
                        ]'];     
                    LL(k) =  modelesGMMseq{ne,K,m}.loglik;
                    idx = [idx ; [ne,K,m]];
                    k=k+1;
                end
            end
        end
    end
    
    figure, subplot(131), plot(C(:,1),C(:,2),'.'),title(sprintf('AIC jeu %s',noms{ii}),'interpreter','latex')
    subplot(132), plot(C(:,1),C(:,3),'.'),title(sprintf('BIC jeu %s',noms{ii}),'interpreter','latex')
    subplot(133), plot(C(:,1),C(:,4),'.'),title(sprintf('ICL jeu %s',noms{ii}),'interpreter','latex')
    set(gcf,'Position',[1,1,1920,954])    
    saveas(gcf,sprintf('criteres_jeu_%s',jeux{ii}),'png')
    
    % select the likeliest model
    lesK = C(:,1);
    u = unique(lesK); u(isinf(u)) = [];
    p = zeros(length(u),1);
    clear bestModel
    for i=1:length(u)
        f = find(lesK==u(i));
        [ll,a] = max(LL(f));
        p(i) = f(a);
        bestModel(i) = modelesGMMseq{idx(p(i),1),idx(p(i),2),idx(p(i),3)};
    end
    % Plot the criteria for each cluster number
    figure(1), clf, hold on, xlabel('Nb clusters'), ylabel('AIC'), grid minor % AIC
    set(gcf,'Position',[1           1        1920         954])
    set(gca,'Fontsize',28)
    
    figure(2), clf, hold on, xlabel('Nb clusters'), ylabel('BIC'), grid minor % BIC
    set(gcf,'Position',[1           1        1920         954])
    set(gca,'Fontsize',28)
    
    figure(3), clf, hold on, xlabel('Nb clusters'), ylabel('ICL'), grid minor % ICL
    set(gcf,'Position',[1           1        1920         954])
    set(gca,'Fontsize',28)
    
    for i=1:length(bestModel)
        figure(1),plot(bestModel(i).nb_clusters,bestModel(i).criteria.AIC,'bo','linewidth',3,'markersize',12)
        figure(2),plot(bestModel(i).nb_clusters,bestModel(i).criteria.BIC,'bo','linewidth',3,'markersize',12)
        figure(3),plot(bestModel(i).nb_clusters,bestModel(i).criteria.ICL,'bo','linewidth',3,'markersize',12)
    end
    saveas(figure(1),['AIC_realData_jeu_' jeux{ii}],'epsc'), saveas(figure(1),['AIC_realData_jeu_' jeux{ii}],'fig'),
    saveas(figure(2),['BIC_realData_jeu_' jeux{ii}],'epsc'), saveas(figure(2),['BIC_realData_jeu_' jeux{ii}],'fig'),
    saveas(figure(3),['ICL_realData_jeu_' jeux{ii}],'epsc'), saveas(figure(3),['ICL_realData_jeu_' jeux{ii}],'fig'),

end
%[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(les_tau,1,25,0,1e-4,0)
%[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(les_tau+1*randn(1,length(les_tau)),1,14,0,1e-4,0)




