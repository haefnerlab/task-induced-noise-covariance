function P = pdf(x, mog_p, discretize)
%MOG.PDF compute pdf of 1d mixture-of-gaussians.
%
%A MoG (in 1d only) is specified by a mean, standard deviation, and weight
%at each mode. A distribution with N modes is represented by a [1 x 3N] row vector:
%
%   mog = [mu_1 sigma_1 pi_1, ..., mu_n sigma_n, pi_n]
%
%To use use vectorization to evaluate many PDFs at once, mog can be a [M x 3N] matrix specifying M
%different MoG distributions. In this case, x should be [M x 1]
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

% Compute PDF
P_per_mode = zeros(size(x,1), size(x,2), N);
for i=1:size(x,2)
    P_per_mode(:,i,:) = normpdf(x(:,i), mus, sigmas);
end
P = sum(P_per_mode .* reshape(pis, M, 1, N), 3);
if nargin > 2 && discretize
    if M > 1
        P = P ./ sum(P, 2);
    else
        P = P ./ sum(P);
    end
end
P = reshape(P, szx);
end