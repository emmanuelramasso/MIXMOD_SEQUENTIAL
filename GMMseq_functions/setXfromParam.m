function x = setXfromParam(modele)

K = modele.nb_clusters;
tau = modele.tau;
beta = modele.beta;
gamma = modele.gamma;

g = sqrt(gamma);
b = sqrt(beta);
xi = tau/double(modele.T);
xi = log(xi ./ (1 - xi));
x=[b(2:K) g(2:K) xi(2:K)]; % be careful to order 1) beta (b) 2) gamma (g) 3) tau (xi)
