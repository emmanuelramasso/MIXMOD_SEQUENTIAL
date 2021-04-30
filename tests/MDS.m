% This script is called in histogrammes_onsets_postClustering_2
% It computes some graphs to show MDS results


% Dissimilarities between means - use Euclidean
%assert(bestModel{end}.initmodel.meth ~= "segments")

in = bestModel{end}.mu;
tau = bestModel{end}.tau;
% dissimilarities = pdist(in,'mahalanobis',squeeze(mean(bestModel{end}.sigma,1)));
dissimilarities = pdist(in);
% dissimilarities = zeros(size(in,1));
% for i=1:size(in,1)
%     for j=i+1:size(in,1)
%         x1 = in(i,:);
%         x2 = in(j,:);
%         dissimilarities(i,j) = pdist([x1; x2],'mahalanobis',...
%             squeeze(0.5*(bestModel{end}.sigma(i,:,:)+bestModel{end}.sigma(j,:,:))));
%         dissimilarities(j,i) = dissimilarities(i,j);
%     end
% end

d = 2;

if 0
    
    % MDS scaling
    [Y,stress,disparities] = mdscale(dissimilarities,d);
    
    % compute distances afterwards
    distances = pdist(Y);
    
    % compare
    [dum,ord] = sortrows([disparities(:) dissimilarities(:)]);
    figure,plot(dissimilarities,distances,'bo', ...
        dissimilarities(ord),disparities(ord),'r.-');
    xlabel('Dissimilarities'); ylabel('Distances/Disparities')
    legend({'Distances' 'Disparities'},'Location','NW');
    title('MDS + Euclidean + Stress normalized by the sum of squares of the inter-point distances')
    
else
    
    % Similar as before, do metric scaling on the same dissimilarities
    figure
    [Y,stress] = mdscale(dissimilarities,d,'criterion','sammon');
    %[Y,stress] = mdscale(dissimilarities,d,'criterion','metricsstress');
    %[Y,stress] = mdscale(dissimilarities,d,'criterion','strain');
    
    distances = pdist(Y);
    plot(dissimilarities,distances,'bo', ...
        [0 max(dissimilarities)],[0 max(dissimilarities)],'r.-');
    xlabel('Dissimilarities'); ylabel('Distances')
    title('MDS + Euclidean + SAMMON')
    
end

% we want to know whether clusters can be "fused" so that close clusters
% have close values of tau_k
figure,subplot(122),plot(Y(:,1),Y(:,2),'bo','MarkerSize',10,'MarkerFaceColor','b'),set(gca,'Fontsize',26)
for i=1:size(in,1), text(Y(i,1)+0.2,Y(i,2)+0.2,sprintf('$\\mu_{%d}$',i),'fontsize',24,'interpreter','latex')
end
grid minor
xlabel('Dim 1 of means after MDS','interpreter','latex','fontsize',30),
ylabel('Dim 2 of means after MDS','interpreter','latex','fontsize',30),
% xlim([-6,10])

subplot(121),plot(tau,'bo','MarkerSize',10,'MarkerFaceColor','b'),set(gca,'Fontsize',26);
for i=1:size(in,1), text(i+0.2,tau(i)+0.2,sprintf('$\\tau_{%d}$',i),'fontsize',24,'interpreter','latex')
end
grid minor
% ylim([-1 65]); xlim([0 15])
xlabel('Onset index','interpreter','latex','fontsize',30),
ylabel('Onsets values ($\tau$)','interpreter','latex','fontsize',30),
set(gcf,'Position',[1           1        1920         954])

figure,plot3(tau,Y(:,1),Y(:,2),'bo'),
for i=1:size(in,1), text(tau(i)+0.2,Y(i,1)+0.2,Y(i,2)+0.2,sprintf('$\\mu_{%d}$',i),'fontsize',24,'interpreter','latex')
end
grid minor

