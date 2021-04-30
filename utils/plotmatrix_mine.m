function h=plotmatrix_mine(sc, lab, creerFig, monoF)

if nargin==2, creerFig=true; monoF=false; end
if nargin==3, monoF=false; end

u = unique(lab);
col=colormap('jet'); %close(gcf)
col = col(2:floor(size(col,1)/length(u)):end,:);

if creerFig, h=figure; else hold on, end

k = 1;
for i=1:size(sc,2)
   for j=1:size(sc,2)
       if ~monoF
           subplot(size(sc,2),size(sc,2),k)
       end
      if i==j,
         if ~monoF
             [a b]=hist(sc(:,i),10); bar(b,a);
         end
      elseif j>i
         for l=1:length(u)
            f = find(lab == u(l));
            hold on
            plot(sc(f,i), sc(f,j), 'o', 'MarkerSize',2, 'Color',col(l,:),'MarkerFaceColor',col(l,:));
         end
      end
      k=k+1;
   end
end
hold off;