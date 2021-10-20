function mog = create(mus, sigmas, pis)
%MOG.CREATE create a mixture-of-gaussians.
%
%A MoG (in 1d only) is specified by a mean, standard deviation, and weight
%at each mode. A distribution with N modes is represented by a [1 x 3N] row vector:
%
%   mog = [mu_1 sigma_1 pi_1, ..., mu_n sigma_n, pi_n]

pis = pis / sum(pis);
mog = [mus(:) sigmas(:), pis(:)]';
mog = mog(:)';

if any(sigmas < 1e-6)
    warning('mog:delta', 'MoG sigma is nearing a delta-distribution. This will cause mog.pdf and mog.logpdf to fail!');
end

end