function mu = mean(mog)
%MOG.MEAN return the overal mean of the given mixture of gaussians

mu = sum(mog(:, 1:3:end) .* mog(:, 3:3:end));

end