function x = sample(mog, S)
%MOG.SAMPLE sample from a 1d mixture of gaussians. If mog is vectorized to [M x 3N] where N is
%number of mixture components and M is number of separate distributions, result is size [M x S]
%where 'S' is requested number of samples.
%
%See MOG.PDF for format.

if nargin < 2, S = 1; end

[M, N3] = size(mog);
N = N3/3;

cumul_modes = reshape(cumsum(mog(:,3:3:end), 2), M, [], N);

% In each row, cumul_modes is the CMF of mode weights. rand > CMF results in a vector of 1s followed
% by 0s where the first zero is the chosen index. For example, if it is [1 1 1 0 0 0 0 0] that
% indicates that the 4th item should be chosen. Hence, the index of the chosen mode is sum(rand <=
% CMF) + 1.
mode = sum(rand(M, S) > cumul_modes, 3) + 1;

% Get mu and sigma for each mode and draw a sample
iMu = (1:M)' + 3*M*(mode-1);
iSig = iMu+M;

x = mog(iMu) + randn(M, S) .* mog(iSig);
end