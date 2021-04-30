function [b,g,xi,beta,gamma,tau] = getParamFromX(x,K,T)

Lambda = @(u) 1.0 ./ (1.0 + exp(-u) );

x = x(:);
kprime = K-1;
b = x(1:kprime); % b first / beta
g = x(kprime+1:2*kprime); % g second / gamma
xi = x(2*kprime+1:3*kprime);% xi then / tau

beta = [1.0; b.^2]';
gamma = [0.0; g.^2]';
tau = [0.0; double(T)*Lambda(xi)]';

