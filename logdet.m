function ld = logdet(A)
l = chol(A);
ld = 2*sum(log(diag(l)));
end