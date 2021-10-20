function L = logpdf(x, mog_p)
%MOG.LOGPDF compute log pdf of 1d mixture-of-gaussians more stably than
%log(mog.pdf())
%
%See mog.pdf for more information.
%
%To use use vectorization to evaluate many PDFs at once, mog can be a size [M, 3N] matrix specifying
%M different MoG distributions. In this case, x should be [M, ...]
%
%Note: unexpected behavior may occur if any mode's sigma is too small, as in a delta distribution

mus = mog_p(:, 1:3:end);
sigmas = mog_p(:, 2:3:end);
pis = mog_p(:, 3:3:end);

% Check if mog is vectorized and adjust input sizes accordingly
[M, N3] = size(mog_p);
N = N3/3;
szx = size(x);
if M > 1
    assert(szx(1) == M, 'Vectorized MOG.LOGPDF dimension mismatch between size of ''x'' and number of distributions');
    x = reshape(x, szx(1), []);
else
    x = x(:);
end

% Compute log PDF
log_mode_probs = zeros(size(x,1), size(x,2), N);
for i=1:size(x,2)
    log_mode_probs(:,i,:) = log(pis) - 1/2*(x(:,i) - mus).^2 ./ sigmas.^2 - 1/2*log(2*pi*sigmas.^2);
end
L = logsumexp(log_mode_probs, 3);
L = reshape(L, szx);
end

function s = logsumexp(a, dim)
% Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
% Default is dim = 1 (columns).
% logsumexp(a, 2) will sum across rows instead of columns.
% Unlike matlab's "sum", it will not switch the summing direction
% if you provide a row vector.

% Written by Tom Minka
% (c) Microsoft Corporation. All rights reserved.

if nargin < 2
  dim = 1;
end

% subtract the largest in each column
[y, ~] = max(a,[],dim);
dims = ones(1,ndims(a));
dims(dim) = size(a,dim);
a = a - repmat(y, dims);
s = y + log(sum(exp(a),dim));
i = find(~isfinite(y));
if ~isempty(i)
  s(i) = y(i);
end
end