function [v2 csca] = reordonne_clusters(bgkfcm, Q, type)

csca = [];
switch type
   case 'temps'
      v=bgkfcm;
      %disp('Pour visu...')
      uu=1:Q;%unique(v);
      h=[];
      for j=1:length(uu)
         ff=find(v==uu(j));
         if length(ff)==0, ff=0; end
         h=[h ff(1)];% contient le premier instant d'apparition
      end
      clear ff
      [u w]=sort(h,'ascend');
      v2=zeros(size(v));
      for j=1:length(w)
         ff=find(v==w(j));
         v2(ff)=j;
      end

   case 'logCSCA'
      
       if 1
           %
           %disp('Reordonne pour visu...')
           uu=1:Q;%unique(v);
           h=[];
           for j=1:length(uu)
               ff=find(bgkfcm==uu(j));% trouve le cluster uu(j)
               if length(ff)==0, ff=0; end
               h=[h ff(1)];% contient le premier instant d'apparition
           end
           clear ff
           [u w]=sort(h,'ascend');% relabel
           v2=zeros(size(bgkfcm));
           for j=1:length(w)
               ff=find(bgkfcm==w(j));
               v2(ff)=j;
           end
           clear ff  u j h uu
           
           if nargout==2
               csca=zeros(length(v2),Q);
               for i=1:length(v2)
                   if v2(i)>0
                       csca(i,v2(i))=1;
                   end
               end
           end
       else
           
%            c=zeros(length(bgkfcm),Q);
%            for i=1:length(bgkfcm)
%                if bgkfcm(i)>0
%                    c(i,bgkfcm(i))=1;
%                else error('??'); end
%            end
%            h = sum(c,1); %h=h(end,:);
%            [u w]=sort(h,'descend');
%            v2=zeros(size(bgkfcm));
%            for j=1:length(w)
%                ff=find(bgkfcm==w(j));
%                v2(ff)=j;
%            end
%            if nargout==2
%                c=zeros(length(bgkfcm),Q);
%                for i=1:length(v2)
%                    c(i,v2(i))=1;
%                end
%                csca = c; clear c
%            end
       end
       
   otherwise error('??');
end