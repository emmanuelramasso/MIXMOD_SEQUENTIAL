function [time_onsets,ARI_est] = apply_clustering_standard(X,temps,method,criterion,setofK,dataCampaign,lesduree,Ytrain)
% call evalclusters(X,method,criterion,'klist',setofK);
% find onsets and plot histogram for common clustering methods

% CHOIX DES CLUSTERS AVEC CRITERES ONSETS
% faire tourner algo pour K=kmin:kmax
% chercher le demarrage de chaque cluster find(first)
time_onsets = []; ARI_est = [];
for k=setofK
        
    if strcmp(method,'gmdistribution')
        options=statset;
        options.MaxIter=1000; % 1 can be enough
        gmfit= fitgmdist(X,k,...
            'CovarianceType','full',...
            'Replicates',10,...
            'options',options); %diagonal
        clusters = cluster(gmfit,X);
            
    else    
        E = evalclusters(X,method,criterion,'klist',k);
        clusters = E.OptimalY;
    end    
    
    ARI_est = [ARI_est valid_RandIndex(clusters,Ytrain)];
    
    if 0
        logcumc = clusters_logcsca(clusters, k);
        figure,plot(logcumc),title('Cumulated nb of clusters')
    end
    % find onsets
    onsets = nan(1,k);
    for lek = 1:k
        f = find(clusters==lek, 1, 'first');
        if not(isempty(f))
            onsets(lek) = f(1);
        end
    end
    time_onsets = [time_onsets ; temps(sort(onsets,'ascend'))];
end

[a,b]=hist(time_onsets,[0:1:max(temps)]);
a = a/sum(a);
ffig=figure; hold on, set(gca,'fontsize',22)
for i=1:length(lesduree)
    d = abs(b-lesduree(i)); [d,e] = min(d);
    line(lesduree(i)*ones(1,10),linspace(a(e),max(a)+0.10*max(a),10),'LineStyle','--','Color',[0 0 0])
end
bar(b,a)
xlabel('Onsets time ($\tau$)','interpreter','latex','fontsize',26),
ylabel('Normalised histogram of onsets','interpreter','latex','fontsize',26)
xlim([-1,65])
title([dataCampaign ' with ' method ' + ' criterion ' with K=' num2str(min(setofK)) ':' num2str(max(setofK))],'interpreter','latex','fontsize',26)
set(gcf,'Position',[1           1        1920         954])
saveas(ffig,['data_' dataCampaign '_' method '_' criterion],'fig')
saveas(ffig,['data_' dataCampaign '_' method '_' criterion],'epsc')
saveas(ffig,['data_' dataCampaign '_' method '_' criterion],'png')

