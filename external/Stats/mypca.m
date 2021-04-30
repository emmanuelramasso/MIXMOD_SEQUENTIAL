function [finaldata,V,D,finaldata1stcomponent,maxeigval]=mypca(x)

numdata = size(x,1);
%step 2, finding a mean and subtracting
xmean=mean(x);
xnew=x-repmat(xmean,numdata,1);

%step 3, covariance matrix
covariancematrix=cov(xnew);

%step 4, Finding Eigenvectors
[V,D] = eig(covariancematrix);
D=diag(D);
maxeigval=V(:,find(D==max(D)));
[D, b]=sort(D,'descend');
V=V(:,b);

%step 5, Deriving the new data set
%finding the projection onto the eigenvectors
finaldata1stcomponent=(maxeigval'*[xnew]')';
finaldata=(V'*xnew')';



