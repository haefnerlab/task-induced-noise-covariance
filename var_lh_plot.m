function fig = var_lh_plot(expt, start_itr, end_itr, SEM, n_plot_zero_sig, n_est_cov, n_avg_post)
% Generate 'caterpillar' plot of noisy likelihoods
if exist('sd', 'var') && ~isempty(sd)
    rng(sd, 'twister');
end

% Default args

% Which iteration # to plot(default 0)
if ~exist('start_itr', 'var') || isempty(start_itr), start_itr = 0; end
% Which iteration # to plot(default 10)
if ~exist('end_itr', 'var') || isempty(end_itr), end_itr = 10; end
% Whether to normalize stdev by sqrt(n) where n is # of times that learning has been run
if ~exist('SEM', 'var') || isempty(SEM), SEM = false; end
% number of zero-signal contours to plot at each iteration
if ~exist('n_plot_zero_sig', 'var') || isempty(n_plot_zero_sig), n_plot_zero_sig = 20; end
% number of posteriors to plot to illustrate the prior
if ~exist('n_avg_post', 'var') || isempty(n_avg_post), n_avg_post = 100; end
% number of zero-signal data points to use when estimating cov-of-p(X) and dp/ds [per iteration]
if ~exist('n_est_cov', 'var') || isempty(n_est_cov), n_est_cov = 1e4; end

xs = linspace(-3, 3, 51);
[xx, yy] = meshgrid(xs);

switch upper(expt)
    case 'SHIFT'
        % Parameterize how the mean of the likelihood depends on s
        mu_x_s = @(s) [s(:) (s(:)+s(:).^3)/10];
        svalues = linspace(-3,3);
        mu_x_values = mu_x_s(svalues);
        
        % Parameterize how the covariance of the likelihood depends on s (it doesn't)
        cov_x_s = @(s) eye(2);
    case 'COVARIANCE'
        % Parameterize how the mean of the likelihood depends on s (it doesnt)
        mu_x_s = @(s) [0 0];
        svalues = linspace(-3,3);
        mu_x_values = mu_x_s(svalues);
        
        % Parameterize how the covariance of the likelihood depends on s
        c = .9;
        covplus = 1.5*[1 c; c 1];
        covzero = eye(2);
        covminus = 1.5*[1 -c; -c 1];
        cov_x_s = @(s) covzero + (s > 0)*(covplus-covzero)*tanh(s) + (s <= 0)*(covminus-covzero)*abs(tanh(s));
    otherwise
        error('Invalid: %s', expt);
end

% Function handles to sample *noisy* likelihoods as a function of s
likelihood_fn_mu = @(s,noise) mu_x_s(s) + randn(1,2).*noise/2;
likelihood_fn_sigma = @(s,noise) cov_x_s(s) + randcov()/10*noise;

%% Load data from start and end of learning

[~, log_prior_start] = load_learned(expt, start_itr, xx);
[prior, log_prior_learned] = load_learned(expt, end_itr, xx);

%% Get some stats

% Estimate dp/ds and variance stats *before* learning
dp_ds_before = estimate_deriv(xx, yy, likelihood_fn_mu, likelihood_fn_sigma, log_prior_start, n_est_cov);
[cov_before, var_vals_before] = est_cov_stats(xx, yy, likelihood_fn_mu, likelihood_fn_sigma, log_prior_start, n_est_cov, dp_ds_before);

% Estimate dp/ds and variance stats *after* learning
dp_ds_after = estimate_deriv(xx, yy, likelihood_fn_mu, likelihood_fn_sigma, log_prior_learned, n_est_cov);
[cov_after, var_vals_after] = est_cov_stats(xx, yy, likelihood_fn_mu, likelihood_fn_sigma, log_prior_learned, n_est_cov, dp_ds_after);

if SEM
    norm = sqrt(size(log_prior_learned, 1));
else
    norm = 1;
end

% Mean +/- SEM of variance along dp/ds
var_along(1,1) = mean(var_vals_before);
var_along(1,2) = std(var_vals_before) / norm;
var_along(2,1) = mean(var_vals_after);
var_along(2,2) = std(var_vals_after) / norm;

% Mean +/- SEM of *fraction of* variance along dp/ds
frac_var_along(1,1) = mean(var_vals_before ./ trace(cov_before));
frac_var_along(1,2) = std(var_vals_before ./ trace(cov_before)) / norm;
frac_var_along(2,1) = mean(var_vals_after ./ trace(cov_after));
frac_var_along(2,2) = std(var_vals_after ./ trace(cov_after)) / norm;

%% Make single figure with relevant stuff after it's all converged
fig = figure;

% Subplot 1: noiseless likelihoods as a fn of s
disp('Plot 1: likelihood fn');
ax = subplot(2,4,1); hold on;
nsplot = 11;
svalues = linspace(-3, 3, nsplot);
splotidx = round(linspace(1, length(svalues), nsplot));
colors = [linspace(0, 1, nsplot); linspace(.8, .64, nsplot); zeros(1, nsplot)]';
for i=1:nsplot
    log_lh = logmvnpdf(xx, yy, likelihood_fn_mu(svalues(splotidx(i)),0), likelihood_fn_sigma(svalues(splotidx(i)),0));
    mycontour(ax, xx, yy, exp(log_lh), 1, '-', colors(i, :));
end
plot(mu_x_values(:,1), mu_x_values(:,2), '-k');
axis(ax, 'square'); xlim(ax, [-3 3]); ylim(ax, [-3 3]);
set(ax, 'xtick', [], 'ytick', []);
title(ax, 'likelihoods, varying s, no noise');
drawnow;

% Subplot 2 & 5: noisy likelihoods & posteriors @ s=0
disp('Plot 2, 5: noise draws @ s=0');
for i=1:n_plot_zero_sig
    % Compute & plot value of LH across grid
    log_lh = logmvnpdf(xx, yy, likelihood_fn_mu(0,1), likelihood_fn_sigma(0,1));
    ax = subplot(2,4,2); hold on;
    mycontour(ax, xx, yy, exp(log_lh), 1, [.9 0 0], [.3 0 0], .2);
    
    % Compute & plot value of posterior
    log_posterior = log_lh + reshape(mean(log_prior_learned, 1), size(xx));
    ax = subplot(2,4,5); hold on;
    mycontour(ax, xx, yy, log2prob(log_posterior), 1, [.8 0 .8], [.2 0 .2], .2);
end
ax = subplot(2,4,2);
axis(ax, 'square'); xlim(ax, [-3 3]); ylim(ax, [-3 3]);
set(ax, 'xtick', [], 'ytick', []);
title(ax, 'noisy likelihoods, s=0');
plot(mu_x_values(:,1), mu_x_values(:,2), '-k');
ax = subplot(2,4,5);
axis(ax, 'square'); xlim(ax, [-3 3]); ylim(ax, [-3 3]);
set(ax, 'xtick', [], 'ytick', []);
plot(mu_x_values(:,1), mu_x_values(:,2), '-k');
title(ax, 'noisy posteriors, s=0');
drawnow;

% Subplot 3-4: prior (set of noisy posteriors @ all s)
disp('Summary plot 3');
ax = subplot(2,4,3); hold on;
svalues = linspace(-3, 3, n_avg_post);
svalues = svalues(randperm(length(svalues)));
for i=1:n_avg_post
    log_lh = logmvnpdf(xx, yy, likelihood_fn_mu(svalues(i),1), likelihood_fn_sigma(svalues(i),1));
    log_posterior = log_lh + reshape(mean(log_prior_learned, 1), size(xx));
    mycontour(ax, xx, yy, log2prob(log_posterior), 1, [0 0 .9], [0 0 .3], .2);
end
axis(ax, 'square'); xlim(ax, [-3 3]); ylim(ax, [-3 3]);
set(ax, 'xtick', [], 'ytick', []);
plot(mu_x_values(:,1), mu_x_values(:,2), '-k');
title(ax, 'posterior samples, all s');
drawnow;

disp('Summary plot 4');
ax = subplot(2,4,4); hold on;
contourf(ax, xx, yy, reshape(prior, size(xx)), 10);
axis(ax, 'square'); xlim(ax, [-3 3]); ylim(ax, [-3 3]);
set(ax, 'xtick', [], 'ytick', []);
plot(mu_x_values(:,1), mu_x_values(:,2), '-k');
title(ax, 'learned prior');
drawnow;

% Subplots 6-7
disp('Summary plot 6, 7');

ax = subplot(2,4,6); hold on;
avg_dp_ds = symmetrize(reshape(mean(dp_ds_after, 1), size(xx)), -1, expt);
contourf(ax, xx, yy, avg_dp_ds, 10);
axis(ax, 'square'); xlim(ax, [-3 3]); ylim(ax, [-3 3]);
set(ax, 'xtick', [], 'ytick', []);
plot(mu_x_values(:,1), mu_x_values(:,2), '-k');
title(ax, 'dp/ds');
drawnow;

ax = subplot(2,4,7); hold on;
bar(ax, [1 2], [var_along(1,1) var_along(2,1)]);
errorbar(ax, [1 2], [var_along(1,1) var_along(2,1)], [var_along(1,2) var_along(2,2)], 'ok');
set(ax, 'XTick', [1 2], 'XTickLabel', {'before learning', 'after learning'});
axis(ax, 'square');
title(ax, 'var. along dp/ds');

ax = subplot(2,4,8); hold on;
bar(ax, [1 2], [frac_var_along(1,1) frac_var_along(2,1)]);
errorbar(ax, [1 2], [frac_var_along(1,1) frac_var_along(2,1)], [frac_var_along(1,2) frac_var_along(2,2)], 'ok');
set(ax, 'XTick', [1 2], 'XTickLabel', {'before learning', 'after learning'});
axis(ax, 'square');
title(ax, 'fraction of var. along dp/ds');
end

%% Helpers

function [prior, log_priors] = load_learned(expt, itr, xx)
if itr == 0
    log_uniform = -log(numel(xx)) * ones(size(xx));
    log_uniform = log_uniform(:)';
    log_priors = log_uniform;
else
    disp(['Pattern: svres-' lower(expt) '*iter=' num2str(itr) '.*']);
    files = dir(fullfile('learning', ['svres-' lower(expt) '*iter=' num2str(itr) '.*']));
    for iF=length(files):-1:1
        fprintf('\tloading %s\n', files(iF).name);
        ld = load(fullfile('learning', files(iF).name));
        log_priors(iF,:) = ld.log_prior(:)';
    end
end

prior = log2prob(mean(log_priors, 1));
end

function xy = symmetrize(xy, sgn, type)
switch upper(type)
    case 'SHIFT'
        % symmetry is xy = -rot180(xy)
        xy = (xy + sgn*rot90(xy,2)) / 2;
    case 'COVARIANCE'
        % symmetries are (1) xy = -flipud(xy) (2) xy = -fliplr(xy) (3) xy = +rot180(xy)
        xy = (xy + rot90(xy, 2) + sgn*flipud(xy) + sgn*fliplr(xy)) / 4;
end
end

function dp_ds = estimate_deriv(xx, yy, lh_fn_mu, lh_fn_sig, log_prior, nsamples)
ds = .05;
nRuns = size(log_prior, 1);
for iSamp=nsamples:-1:1
    if mod(iSamp, 1000) == 0, fprintf('dp/ds-est: iter %d / %d\n', iSamp, nsamples); end
    % Get LH at ±ds with the same random seed each time (reparameterization trick, essentially)
    tmp_rng_state = rng();
    lh_log_prob_pos = logmvnpdf(xx, yy, lh_fn_mu(+ds,1), lh_fn_sig(+ds,1));
    rng(tmp_rng_state);
    lh_log_prob_neg = logmvnpdf(xx, yy, lh_fn_mu(-ds,1), lh_fn_sig(-ds,1));
    for iRun=1:nRuns
        smpl_dp_ds(iRun, :, iSamp) = log2prob(exp(log_prior(iRun,:)' + lh_log_prob_pos(:))) - ...
            log2prob(exp(log_prior(iRun,:)' + lh_log_prob_neg(:)));
    end
end
dp_ds = mean(smpl_dp_ds, 3);
end

function [post_cov, var_post_vals, fi_post_vals] = est_cov_stats(xx, yy, lh_fn_mu, lh_fn_sig, log_priors, nsamples, dp_ds)
post_cov = zeros(numel(xx));
unit_dp_ds = dp_ds ./ sqrt(sum(dp_ds.^2, 2));
nRuns = size(log_priors, 1);
for iPri=nRuns:-1:1
    for iSamp=nsamples:-1:1
        if mod(iSamp, 1000) == 0, fprintf('cov-est: pri %d /%d\titer %d / %d\n', iPri, nRuns, iSamp, nsamples); end
        
        % Compute value of LH across grid at s=0
        log_like = logmvnpdf(xx, yy, lh_fn_mu(0,1), lh_fn_sig(0,1));
        
        % Compute posteriors
        log_post = log_like(:)' + log_priors(iPri, :);
        
        % Add a row to LH and Posterior instances
        post_prob(iSamp, :)  = log2prob(log_post);
    end
    
    this_post_cov = cov(post_prob);
    % Get variance of projection onto unit dp_ds diretion
    var_post_vals(iPri) = var(post_prob * unit_dp_ds(iPri, :)');
    % Approximate fisher information as ||dp/ds||^2 / variance along dp/ds
    fi_post_vals(iPri) = norm(dp_ds(iPri, :), 2)^2 / var_post_vals(iPri);
    % Average covariance of posterior density, averaged across learning runs
    post_cov = post_cov + this_post_cov;
end

post_cov = post_cov / nRuns;
end

function h = mycontour(ax, xx, yy, zz, n, patchcolor, linecolor, alpha)
%Create lines or transpartent patches at n contour levels
tmp = figure;
[c, h] = contour(xx, yy, zz, n);
close(tmp);

hold(ax, 'on');
idxstart = 1;
for i=1:n
    numpoints = c(2,idxstart);
    xs = c(1, idxstart+1:idxstart+numpoints);
    ys = c(2, idxstart+1:idxstart+numpoints);
    if ischar(patchcolor) && strcmp(patchcolor, '-')
        h(i) = plot(ax, xs, ys, 'Color', linecolor);
    else
        h(i) = patch(xs, ys, patchcolor);
        h(i).EdgeColor = linecolor;
        h(i).FaceAlpha = alpha;
    end
    idxstart = idxstart + numpoints + 1;
end
end

function C = randcov()
%Create a random covariance matrix by drawing random marginal deviations (s1,s2) and a random
%correlation (c)
s1 = exprnd(1);
s2 = exprnd(1);
c = tanh(randn(1));
C = [1 c; c 1] .* ([s1 s2]' .* [s1 s2]);
end

function log_probs = logmvnpdf(xx, yy, mu, sigma)
%Evaluate the 2D log probability of (x,y) for the 2D gaussian with mean mu and covariance sigma
xy = [xx(:) yy(:)];
log_probs = -1/2 * sum((xy - mu)' .* (sigma \ (xy - mu)'), 1) -1/2 * logdet(sigma);
log_probs = reshape(log_probs, size(xx));
end

function p = log2prob(log_prob)
p = exp(log_prob(:)-max(log_prob(:)));
p = p / sum(p);
p = reshape(p, size(log_prob));
end