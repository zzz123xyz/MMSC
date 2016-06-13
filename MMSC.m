function [G, obj_value] = MMSC(data, nbclusters, a, method, param)
%%
%Cai, Xiao, et al. "Heterogeneous image feature integration via multi-modal
%spectral clustering." Computer Vision and Pattern Recognition (CVPR),
%2011 IEEE Conference on. IEEE, 2011.

% by Lance Liu 26/05/16

%input
% data: for k views of features, k row cells with R^{d \times n} in each
% cell
% nbclusters: number of clusters
% a: penalty parameter
% method: method used to compute the affinity matrix
% para: parameter for the used method.

%output
% ???

%%
niters = 2;
V = numel(data);
[nfeat, ndata_vec] = cellfun(@size, data); n = ndata_vec(1);

L_mm = zeros(n);
for i = 1:V
    W{i} = constructGraph(data{i}, nbclusters, method, param, 'unormalized');
    D{i} = diag(sum(W{i}, 2));
    L{i} = D{i}- W{i};

    L_mm = L_mm + (L{i}+a*eye(n))^(-1);
end

[eigvec, eigval] = eigs(L_mm, nbclusters,'sm');

sq_sum = sqrt(sum(eigvec.*eigvec, 2)) + 1e-20;
T = eigvec ./ repmat(sq_sum, 1, nbclusters);

G0 = kmeans(T, nbclusters);
G = G0 + 0.2;

%iterations
J = L_mm;
a = G'*J*G; %may cuz problem?
for k = 1:niters
   G = G.*((J*G)./(G*a)).^(0.5);
   obj_value(k) = trace(G'*J*G);
end

G = kmeans(G, nbclusters);
