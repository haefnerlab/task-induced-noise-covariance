function prod = prod(d1, d2)
%MOG.PROD compute product of two 1d mixture-of-gaussian distributions.
%
%See MOG.PDF for format.
%
%To use vectorized version, both d1 and d2 can be [M x 3N], as long as M is the same for both.

modes1 = size(d1, 2) / 3;
modes2 = size(d2, 2) / 3;
modes_out = modes1 * modes2;

M1 = size(d1, 1);
M2 = size(d2, 1);
assert(M1 == M2);

prod = zeros(M1, modes_out*3);
k = 1;
for i=1:modes1
    iMode = 3*(i-1);
    for j=1:modes2
        jMode = 3*(j-1);
        mu_i = d1(:, iMode+1); var_i = d1(:, iMode+2).^2; pi_i = d1(:, iMode+3);
        mu_j = d2(:, jMode+1); var_j = d2(:, jMode+2).^2; pi_j = d2(:, jMode+3);
        mu_k = (var_i .* mu_j + var_j .* mu_i) ./ (var_i + var_j);
        var_k = (var_i .* var_j) ./ (var_i + var_j);
        pi_k = pi_i .* pi_j .* normpdf(mu_i, mu_j, sqrt(var_i + var_j));
        
        kMode = 3*(k-1);
        prod(:, kMode+(1:3)) = [mu_k, sqrt(var_k), pi_k];
        k = k+1;
    end
end
prod_pis = prod(:, 3:3:end);
prod(:, 3:3:end) = prod_pis ./ sum(prod_pis, 2);
end