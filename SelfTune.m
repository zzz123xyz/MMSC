function A = SelfTune(data, k)
%%
% By Lance May 16
% interface to incroprate the CLR algorithm with the graph construction
% stage.

% Input:
%   data: X feature matrix dim: R^{m*n} (m features & n samples)
%   k: k nearest neighbors
%
% Output: 
%   A: the output graph

%% compute sigmma for each point
PairDist = pdist2(data', data');
[P, Idx] = sort(PairDist,2);
KnnIdx = Idx(:,k);
sigmma = P(:,k);

%% 
denominator = kron(sigmma, sigmma');
A = exp(-PairDist.^2 ./ denominator);