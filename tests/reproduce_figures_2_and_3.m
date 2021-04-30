% Script that reproduce figures 2 and 3 of simulated data of our paper
% 
% Must be run in "main_test_simulated_data.m", after data generation 
% Its creates a directory in which images are saved to see evolution of
% clusters.
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



% Make surfaces for each cluster
T = max(temps);

% prepare meshes
ax1mn = min(X(:,1));
ax2mn = min(X(:,2));
ax1mx = max(X(:,1));
ax2mx = max(X(:,2));
x1l = linspace(ax1mn,ax1mx,100);
x2l = linspace(ax2mn,ax2mx,100);
[x1,x2] = meshgrid(x1l,x2l);
contr = zeros([size(x1),K]);
t_repT = temps*ones(1,K);% t on 1...K replicated over T

% compute probs
for i=1:size(x1,1)
    for j=1:size(x1,2)
        
        data = [x1(i,j),x2(i,j)];
        
        % compute y
        phi = zeros(1,K);
        for k=1:K
            if not(true_model.useExternalFunctions)
                phi(:,k) = mvnpdf(data, true_model.mu(k,:), squeeze(true_model.sigma(k,:,:)));
            else
                phi(:,k) = mixgauss_prob(data', true_model.mu(k,:)', squeeze(true_model.sigma(k,:,:)));
            end
        end
        contr(i,j,:) = phi; % juste gaussiennes
        
    end
end
figure,c=contourslice(contr,[],[],[1:K]);
for i=1:length(c), c(i).LineWidth = 2; end
set(gca,'fontsize',22)
xlabel('X_1'),ylabel('X_2')
grid minor
if saveImages
    saveas(gcf,'simu_X1X2','fig')
    saveas(gcf,'simu_X1X2','epsc')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PAS = 30; % to avoid plotting all points
IdxtempsReduit = [1:PAS:length(temps)];

%     % conditional probability
%     [x1,x2,z] = meshgrid(x1l,x2l,IdxtempsReduit);
%     volumeData = zeros(size(x1));
%     for k=1:K
%         for t=1:length(IdxtempsReduit)
%             % slice, conditional probability
%             volumeData(:,:,t) = (pitk(IdxtempsReduit(t),k) .* contr(:,:,k));
%         end
%     end

volumeData = zeros([size(x1),length(IdxtempsReduit)]);
[x1,x2,z] = meshgrid(x1l,x2l,IdxtempsReduit);
for k=1:K
    for t=1:length(IdxtempsReduit)
        % slice, joint distribution
        volumeData(:,:,t,k) = (pitk(IdxtempsReduit(t),k) .* contr(:,:,k));
    end
end

% Plot the clusters in 2D
v = sum(volumeData,4); % sum over k, mixture distribution

% plot activations
activations = figure; ax_activations = axes;
plot(ax_activations,temps,pitk), set(gca,'fontsize',22),
grid(ax_activations,'minor')
ylabel('$\pi_{tk}$ (true model)','interpreter','latex'), xlabel('Time','interpreter','latex')
s = {}; for k=1:K, s{end+1} = sprintf('Cluster %d',k); end
legend(s), set(activations,'Position',[ 104         163        1095         772])

% plot clusters
distrib = figure; ax_distrib = axes;
set(distrib,'Position',[792          45        1095         772])
set(ax_distrib,'fontsize',22)
mkdir('tmpImages')
for t=1:length(IdxtempsReduit)
    cla(ax_distrib)
    colorbar(ax_distrib)
    c = contour(ax_distrib,squeeze(v(:,:,t)),20);
    grid(ax_distrib,'minor')
    title(ax_distrib,sprintf('t=%.1f',temps(IdxtempsReduit(t))),'fontsize',24,'interpreter','latex');
    set(ax_distrib,'fontsize',22)
    xlabel(ax_distrib,'$X_1$','interpreter','latex'),ylabel(ax_distrib,'$X_2$','interpreter','latex')
    drawnow
    if saveImages 
        saveas(distrib,sprintf('tmpImages/cluster_image_%05d',t),'png')
    end
    
    h=line(get(activations,'CurrentAxes'),temps(IdxtempsReduit(t))*ones(1,10),linspace(0,max(pitk(IdxtempsReduit(t),:)),10));
    title(ax_activations,sprintf('t=%.1f',temps(IdxtempsReduit(t))),'fontsize',24,'interpreter','latex');
    drawnow
    if saveImages 
        saveas(activations,sprintf('tmpImages/activ_image_%05d',t),'png')
    end
    delete(h)
    
end
% On linux go to tmpImages and execute to get a gif:
%convert -delay 10 -loop 0 cluster_image_*.png simu_clusters.gif
%convert -delay 10 -loop 0 activ_image_*.png simu_activ.gif

% Plot the clusters in 3D
%     figure
%     for t=1:length(IdxtempsReduit)
%         clf
%         c = contourslice(squeeze(volumeData(:,:,t,:)),[],[],[1:K]);
%         for i=1:length(c), c(i).LineWidth = 2; end
%         colorbar
%         zlim([1,K]), zticks(1:K)
%         title(sprintf('t=%d',IdxtempsReduit(t)));
%         view([-37.5 30]);
%         set(gca,'fontsize',22)
%         grid minor
%         xlabel('X_1'),ylabel('X_2'),zlabel('Cluster'),
%         set(gcf,'Position',[661         149        1095         772])
%         saveas(gcf,sprintf('image_%05d',t),'png')
%     end
%convert -delay 10 -loop 0 image_*.png simu1.gif
