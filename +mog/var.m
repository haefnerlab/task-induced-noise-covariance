function sig2 = var(m)
%MOG.VAR return the overal variance of the given mixture of gaussians

mus = m(:, 1:3:end);
sig2s = m(:, 2:3:end).^2;
pis = m(:, 3:3:end);

moment2 = mus.^2 + sig2s;
sig2 = sum(moment2 .* pis, 2) - mog.mean(m).^2;

end