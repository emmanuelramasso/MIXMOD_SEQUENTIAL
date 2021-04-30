function test_check_likelihood_contour(true_model,X,temps,clusterChoisi,meshstep,saveImages)
% We fix all parameters except one in beta and tau (for a given cluster) 
% in order to plot the likelhood function L(X,time,beta(selected_cluster),tau(selected_cluster)) 
% where gamma is fixed as well as beta and tau for all other clusters. 
% We generate a dataset with the model provided (true_model) from which we
% compute the likelihood of X (with time provided).
% 
% * true_model is a struture with the following fields (with K the 
% number of cluster, and d the dimension of the data)
% - gamma [1,K]
% - beta [1,K]
% - tau [1,K]
% - mu [K,d]
% - sigma [K,d,d]
% * X is the data [L,d]
% * temps is the vector of timestamps of data [L,1]
% * clusterChoisi is the cluster number between 2 and K
% * grid size meshstep x meshstep
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
%

% On fixe tous les paramètres sauf gamma et beta.
% On génère un dataset avec paramètres connus.
% On trace les contours de la vraissemblance évaluée en des couples (gamma,
% beta) et on vérifie si le maximum est bien là où on l'attend

assert(clusterChoisi>1); % no sense to use the first cluster 

K = size(true_model.mu,1);

% on va enlever 1 valeur dans beta, et une dans tau, toutes les autres
% seront gardées fixes
% on choisit un cluster et on modifie 2 de ses paramètres
% on memorise les valeurs des vraies paramètres pour les tracer ensuite
trueValues = [true_model.beta(clusterChoisi),true_model.tau(clusterChoisi)];

% % SETTINGS WITH TRUE MODEL
% listeBeta = linspace(true_model.beta(clusterChoisi)*0.95,...
%     true_model.beta(clusterChoisi)*1.05,meshstep);
% % on génère les valeurs possibles du tau qu'on veut modifier
% listeTau = linspace(true_model.tau(clusterChoisi)*0.95,...
%     min(true_model.tau(clusterChoisi)*1.05,true_model.T),meshstep);
% SETTINGS WITH MLE
listeBeta = linspace(true_model.beta(clusterChoisi)*0.01,...
    true_model.beta(clusterChoisi)*2.5,meshstep);
% on génère les valeurs possibles du tau qu'on veut modifier
listeTau = linspace(true_model.tau(clusterChoisi)*0.5,...
      min(true_model.tau(clusterChoisi)*1.5,true_model.T),meshstep);
%listeTau = linspace(0,true_model.T*0.99,meshstep);

% mesh
[BETA,TAU] = meshgrid(listeBeta,listeTau);
LL = zeros(size(BETA));

fprintf('Mesh %dx%d likelihood...\n',size(BETA));
for i = 1:size(BETA,1)
    
    for j = 1:size(BETA,2)
        
        % on fait varier beta et tau pour un des clusters
        % on doit recalculer les variables instrumentales
        %b(clusterChoisi) = sqrt(BETA(i,j));
        %xitmp = TAU(i,j)/T;
        %xitmp = log(xitmp ./ (1 - xitmp));
        %xi(clusterChoisi) = xitmp;
        
        m = true_model; 
        m.tau(clusterChoisi) = TAU(i,j);
        m.beta(clusterChoisi) = BETA(i,j);
        x = setXfromParam(m);
        %x=[b(2:K) g(2:K) xi(2:K)]'; % be careful to order 1) beta (b) 2) gamma (g) 3) tau (xi)
        
        % on récupère la vraisemblance
        LL0 = GMMSEQ_loglikelihood(x, true_model.mu, true_model.sigma, K, true_model.T, X, temps);
        
        if ~isreal(LL0)
            error('Probablement la valeur de tau n''est pas correct (<temps(max))')
        end
        
        LL(i,j) = LL0;
        
        if mod(i*size(BETA,2)+j,round(10/100*prod(size(BETA))))==0
            fprintf('Currently %dx%d done\n',i,j)
        end
    end
end
fprintf('Currently %dx%d done\n',i,j)
fprintf('ok\n')

mx=max(LL(:)); mn=min(LL(:)); 
n = 0.10;
%q=sort([linspace(mn,mx+n*mx,50) linspace(mx+n*mx,mx,50) linspace(mx+n/10*mx,mx,50)]);
%figure; contourf(BETA,TAU,LL,q); 
figure; [M,c]=contour(BETA,TAU,LL,200); %,'ShowText','on'); 
c.Fill='on';
colormap jet
[a,b]=max(LL(:)); bb=BETA(:); tt=TAU(:); 
% premiers tracés pour legend "noire"
hold on, plot(trueValues(1),trueValues(2),'*','markersize',18,'linewidth',3)
% hold on, plot(bb(b),tt(b),'ko','markersize',18,'linewidth',3)
% second tracés plus visibles - par dessus
% hold on, plot(trueValues(1),trueValues(2),'*','markersize',18,'color',[0.0 0.0 0.0],'linewidth',3)
% hold on, plot(bb(b),tt(b),'o','markersize',18,'color',[0.0 0.0 0.0],'linewidth',3)
legend({'Contours',...
    sprintf('MLE (\\beta=%f,\\tau=%f)',...
    true_model.beta(clusterChoisi),true_model.tau(clusterChoisi))});%, ...
    %sprintf('Maximum of likelihood on grid for (\\beta=%f,\\tau=%f)',...
    %bb(b),tt(b))},'Fontsize',18);
set(gca,'Fontsize',22) 
set(gcf,'Position',[1           1        1920         954])
xlabel(sprintf('$\\beta_%d$',clusterChoisi),'interpreter','latex','Fontsize',26)
ylabel(sprintf('$\\tau_%d$',clusterChoisi),'interpreter','latex','Fontsize',26)

if saveImages
    saveas(gcf,['loglikCheck_cluster' num2str(clusterChoisi)],'epsc')
    saveas(gcf,['loglikCheck_cluster' num2str(clusterChoisi)],'fig')
end

% figure; 
% contourf(BETA,TAU,LL,20);
% % [M,c]=contour(BETA,TAU,LL,'ShowText','on');
% % c.LineWidth=3;
% hold on, plot(trueValues(1),trueValues(2),'*','markersize',15)
% text(trueValues(1)+5/100*trueValues(1),trueValues(2),sprintf('Theoretical maximum of\n the likelihood function'),'interpreter','latex')
% % colormap('cool'), colorbar, grid on
% legend('loglik','True')
% title([sprintf('Contour of the likelihood function: all but one (for cluster %d) are fixed values for',clusterChoisi) ' $\beta$ and $\tau$'],'interpreter','latex')
% disp('The cross should be at the maximum')

% figure; s=surf(BETA,TAU,LL,'FaceAlpha',0.7);
% shading interp
% s.EdgeColor = 'none';
% b(clusterChoisi) = sqrt(trueValues(1));
% xitmp = 1/temps(end)*trueValues(2);
% xitmp = log(xitmp ./ (1 - xitmp));
% xi(clusterChoisi) = xitmp;
% x=[b(2:K) g(2:K) xi(2:K)]'; %
% LLm = GMMSEQ_loglikelihood(x, true_model.mu, true_model.sigma, K, max(temps), L, X, temps);
% hold on, plot3(trueValues(1),trueValues(2),LLm,'*','markersize',15,'linewidth',3,'color',[0 0 0])
% text(trueValues(1)+5/100*trueValues(1),trueValues(2),LLm,sprintf('Theoretical maximum of\n the likelihood function'),'interpreter','latex','fontsize',18)
% colormap('cool'), colorbar, grid on
% legend('loglik','True')
% title(['Contour of the likelihood function by varying $\beta_' num2str(clusterChoisi) '$ and $\tau_' num2str(clusterChoisi) '$'],'interpreter','latex','fontsize',18)
% xlabel(sprintf('$\\beta_%d$',clusterChoisi),'interpreter','latex','fontsize',18)
% ylabel(sprintf('$\\tau_%d$',clusterChoisi),'interpreter','latex','fontsize',18)
% zlabel('Likelihood','fontsize',18,'interpreter','latex')
% disp('The cross should be at the maximum')
% plt = Plot(); % create a Plot object and grab the current figure
% plt.XLabel = 'Sample number'; % xlabel
% plt.YLabel = 'Voltage (mV)'; %ylabel

