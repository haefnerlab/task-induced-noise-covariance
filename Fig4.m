function fig = Fig4(varargin)
p = inputParser;
p.addRequired('expt');  % experiment - one of 'shift' or 'covariance'
p.addParameter('sd', 15647841);  % random seed chosen by keyboard-mashing
p.addParameter('eps', 5e-4); % stop learning when TVD(prior_t,prior_t-1) < eps
p.addParameter('max_iterations', 1000); % ...or stop when max is hit
p.addParameter('n_marginal_s', 200); % resolution of p(s) which we are trying to match
p.addParameter('samples_per_derivative', 1000); % number of noise draws to estimate dp/ds
p.addParameter('variance_samples', 2000); % number of noise draws to estimate %variance
p.addParameter('n_plot_s', 11); % 1st plot panel: how many values of s to show
p.addParameter('n_contours', 20); % 2nd and 3rd plot panel: how many sampled likelihoods/posteriors to show
p.addParameter('analyze_every', 10); % Do dp/ds analysis on iterations (1+k*analyze_every), and final iteration
p.addParameter('lr', 0.01); % (constant) learning rate
p.addParameter('symmetry', true); % whether to enforce rotational symmetries of prior and dp/ds (makes it look nicer and reduces noise in plots)
p.addParameter('varplot', 'bar'); % or 'lines'
p.addParameter('savedir', '.');
p.addParameter('verbose', false);
p.addParameter('debug_plot', false);
p.parse(varargin{:});
args = p.Results;

% Note: this ratio results in roughly equal fraction of variance attributed to ∆µ and ∆Σ before learning
cov_noise_factor = 1/10;
mu_noise_factor = 1/10;

rng(args.sd, 'twister');
sd_sequence = randi(2^31, args.max_iterations+1, 2);

xvalues = linspace(-3, 3, 51);
[xx, yy] = meshgrid(xvalues);


%% Define mu(s) and cov(s) functions
switch upper(args.expt)
    case 'SHIFT'
        % Parameterize how the mean of the likelihood depends on s (cubic, plus noise)
        rand_mean_fn = @(s, noise) [s(:), (s(:)+s(:).^3)/10] + randn(1,2)*noise*mu_noise_factor;
        % Parameterize how the covariance of the likelihood depends on s (constant plus noise)
        rand_cov_fn = @(s, noise) 0.5*eye(2) + randcov(2)*noise*cov_noise_factor;
        
        prior_symmetry = @(p) (p + rot90(p,2))/2;
        dp_ds_symmetry = @(dpds) (dpds - rot90(dpds,2))/2;
        
        % Create true p(s) as a uniform distribution on [-3,3]
        marginal_s_values = linspace(-3, 3, args.n_marginal_s);
        true_marginal_s = normpdf(marginal_s_values);
        true_marginal_s = true_marginal_s / sum(true_marginal_s);
    case 'COVARIANCE'
        % Parameterize how the mean of the likelihood depends on s (constant plus noise)
        rand_mean_fn = @(s, noise) [0 0] + randn(1,2)*noise*mu_noise_factor;
        % Parameterize how the covariance of the likelihood depends on s (transition from + to - corr, plus noise)
        max_correlation = .9;
        covplus = 1.5*[1 max_correlation; max_correlation 1];
        covzero = 0.5*eye(2);
        covminus = 1.5*[1 -max_correlation; -max_correlation 1];
        rand_cov_fn = @(s, noise) covzero + (s>0)*(covplus-covzero)*tanh(s) + (s<0)*(covminus-covzero)*abs(tanh(s)) + randcov(2)*noise*cov_noise_factor;
        
        prior_symmetry = @(p) (p + rot90(p,1) + rot90(p,2) + rot90(p,3))/4;
        dp_ds_symmetry = @(dpds) (dpds - rot90(dpds,1) + rot90(dpds,2) - rot90(dpds,3))/4;
        
        % Create true p(s) proportional to a normal distribution on [-3,3]
        marginal_s_values = linspace(-3, 3, args.n_marginal_s);
        true_marginal_s = normpdf(marginal_s_values);
        true_marginal_s = true_marginal_s / sum(true_marginal_s);
    case 'ROT-SHIFT'
        % Parameterize how the mean of the likelihood depends on s (around a ring, plus noise)
        rand_mean_fn = @(s, noise) 1.5*[cos(pi*s), sin(pi*s)] + randn(1,2)*noise*mu_noise_factor;
        % Parameterize how the covariance of the likelihood depends on s (constant plus noise)
        rand_cov_fn = @(s, noise) 0.5*eye(2) + randcov(2)*noise*cov_noise_factor;
        
        prior_symmetry = @(p) (p + rot90(p,1) + rot90(p,2) + rot90(p,3))/4;
        dp_ds_symmetry = @(dpds) (dpds - flipud(dpds))/2;
        
        % Create true p(s) uniform on [-1,1]
        marginal_s_values = linspace(-1, 1, args.n_marginal_s);
        true_marginal_s = ones(size(marginal_s_values))/args.n_marginal_s;
    case 'ROT-COVARIANCE'
        % Parameterize how the mean of the likelihood depends on s (constant plus noise)
        rand_mean_fn = @(s, noise) [0 0] + randn(1,2)*noise*mu_noise_factor;
        % Parameterize how the covariance of the likelihood depends on s (elongated, pointing to s along the ring)
        rand_cov_fn = @(s, noise) rot(pi/2*s)'*[2 0; 0 0.25]*rot(pi/2*s) + randcov(2)*noise*cov_noise_factor;
        
        prior_symmetry = @(p) (p + rot90(p,1) + rot90(p,2) + rot90(p,3))/4;
        dp_ds_symmetry = @(dpds) (dpds - flipud(dpds))/2;
        
        % Create true p(s) uniform on [-1,1]
        marginal_s_values = linspace(-1, 1, args.n_marginal_s);
        true_marginal_s = ones(size(marginal_s_values))/args.n_marginal_s;
    otherwise
        error('Need EXPERIMENT to be ''SHIFT'' or ''COVARIANCE''.');
end

if args.debug_plot
    figure;
    ax = gca;
    n_vis = 20;
    % green --> orange fade
    s_colors = [linspace(0, 1, n_vis); linspace(.8, .64, n_vis); zeros(1, n_vis)]';
    idx = round(linspace(1, args.n_marginal_s, n_vis+2));
    s_values = marginal_s_values(idx(2:end-1));
    hold on;
    for i=1:n_vis
        log_like = logmvnpdf(xx, yy, rand_mean_fn(s_values(i), 0), rand_cov_fn(s_values(i), 0));
        mycontour(ax, xx, yy, log2prob(log_like), 1, '-', s_colors(i,:));
    end
    if strcmpi(args.expt, 'SHIFT')
        % Underlay the cubic mean function
        uistack(plot(marginal_s_values, (marginal_s_values(:)+marginal_s_values(:).^3)/10, '-k'), 'bottom');
    end
    xlim(ax, [min(xvalues) max(xvalues)]);
    ylim(ax, [min(xvalues) max(xvalues)]);
    axis square;
    title('Parameterization w/out noise');
end

%% Run analysis or load from file if available
save_file = fullfile(args.savedir, sprintf('Fig4_%s.mat', lower(args.expt)));
if exist(save_file, 'file')
    data = load(save_file);
    priors_history = data.priors_history;
    itr_history = data.itr_history;
    dp_ds = data.dp_ds;
    total_variance = data.total_variance;
    var_along_dpds = data.var_along_dpds;
    mass_covariance = data.mass_covariance;
    s_given_x_table = data.s_given_x_table;
    if args.verbose
        fprintf('Loaded precomputed results from %s with %d learning iterations\n', save_file, max(itr_history));
    end
else
    %% Iterate learning by moving the prior such that the marginal likelihood gets closer to the true p(s)
    % Initialize prior on x to uniform
    prior = ones(size(xx)) / numel(xx);
    priors_history = {};
    itr_history = [];
    if args.verbose, fprintf('Precomputing s|x table...\n'); end
    s_given_x_table = likelihood(xx, yy, rand_mean_fn, rand_cov_fn, marginal_s_values, 1000);
    for itr=1:args.max_iterations+1
        if mod(itr-1, args.analyze_every) == 0
            priors_history{end+1} = prior;
            itr_history(end+1) = itr-1;
        end
        
        [err, prior] = learning_step(s_given_x_table, prior, true_marginal_s, args.lr);
        
        if args.symmetry
            prior = prior_symmetry(prior);
        end
        
        if itr >= 2
            delta = tvd(last_prior, prior);
            if args.verbose
                fprintf('Learning itr %03d\tERR = %f\tTVD change in prior = %f\n', itr-1, err, delta);
            end
            
            if delta < args.eps
                if args.verbose
                    fprintf('Breaking after %d learning iterations\n', itr-1);
                end
                priors_history{end+1} = prior;
                itr_history(end+1) = itr;
                break
            end
        end
        last_prior = prior;
    end
    
    if args.debug_plot
        figure;
        for i=1:length(priors_history)
            subplotsquare(length(priors_history), i);
            contourf(xx, yy, priors_history{i}, 20);
            axis square;
            xticks([]); yticks([]);
        end
    end
    if args.verbose && delta > args.eps
        fprintf('Failed to converge at eps=%.5f after %d iterations\n', args.eps, args.max_iterations);
    end
    
    %% Estimate E_{noise}[dp/ds] at s=0, once for each learning iteration
    for i=length(priors_history):-1:1
        rng(sd_sequence(itr_history(i)+1, 1), 'twister');
        dp_ds{i} = estimate_dp_ds(xx, yy, rand_mean_fn, rand_cov_fn, priors_history{i}, args.samples_per_derivative);
        
        if args.symmetry
            dp_ds{i} = dp_ds_symmetry(dp_ds{i});
        end
        
        if args.verbose
            fprintf('dp/ds itr %03d\n', itr_history(i));
        end
        
        if args.debug_plot
            subplotsquare(length(priors_history), i);
            contourf(xx, yy, dp_ds{i}, 20);
            axis square;
            xticks([]); yticks([]);
            drawnow;
        end
    end
    
    %% Estimate total variance and variance in dp/ds direction after each iteration
    for i=length(priors_history):-1:1
        rng(sd_sequence(itr_history(i)+1, 2), 'twister');
        posterior_instances = zeros(args.variance_samples, numel(xx));
        log_pri = log(priors_history{i});
        parfor j=1:args.variance_samples
            % New random posterior at s=0 using prior at iteration itr.
            rand_log_post = log_pri + logmvnpdf(xx, yy, rand_mean_fn(0, 1), rand_cov_fn(0, 1));
            posterior_instances(j,:) = log2prob(rand_log_post(:));
        end
        
        % Get total variance and projected variance
        mass_covariance{i} = cov(posterior_instances);
        total_variance(i) = trace(mass_covariance{i});
        norm_dp_ds = dp_ds{i}(:) / sqrt(sum(dp_ds{i}(:).^2));
        var_along_dpds(i) = var(posterior_instances * norm_dp_ds);
        
        if args.verbose
            fprintf('variance itr %03d\tvar = %f\tdp/ds var = %f\tfrac var = %f\n', itr_history(i), total_variance(i), var_along_dpds(i), var_along_dpds(i)/total_variance(i));
        end
    end
    
    %% Cache all workspace variables to a file
    clear posterior_instances;
    save(save_file);
end

%% (Debugging) marginal likelihood plots
if args.debug_plot
    if strcmpi(args.expt, 'SHIFT')
        demo_x = [-1  0  1 -1 0 1 -1 0 1]*2;
        demo_y = [-1 -1 -1  0 0 0  1 1 1]*2;
    else
        % randomize xy points since there are so many symmetries (quasi-random grid)
        demo_x = linspace(-2, 2, 9);
        demo_y = demo_x(randperm(9));
    end
    figure();
    subplot(1,3,1); hold on;
    colors = hsv(length(demo_x));
    for i=1:length(demo_x)
        plot(demo_x(i), demo_y(i), 'o', 'color', colors(i,:), 'markerfacecolor', colors(i,:));
    end
    xlim([min(xx(:)), max(xx(:))]); ylim([min(yy(:)), max(yy(:))]);
    if strcmpi(args.expt, 'SHIFT')
        uistack(plot(marginal_s_values, (marginal_s_values(:)+marginal_s_values(:).^3)/10, '-k'), 'bottom');
    end
    axis square;
    title('values of x in the second subplot');
    subplot(1,3,2); hold on;
    for i=1:length(demo_x)
        plot(marginal_s_values, likelihood(demo_x(i), demo_y(i), rand_mean_fn, rand_cov_fn, marginal_s_values, 100), 'color', colors(i,:));
    end
    title('p(s|x) for different values of x');
    subplot(1,3,3); hold on;
    plot(marginal_s_values, true_marginal_s, '-k', 'displayname', 'true');
    for i=1:length(priors_history)
        marginal_s = s_given_x_table * priors_history{i}(:);
        plot(marginal_s_values, marginal_s, 'displayname', sprintf('itr %d', itr_history(i)));
    end
    title('marginal p_b(s) over learning');
    legend();
end
%% (Debugging) visualize generative model stuff
if args.debug_plot
    if any(strcmpi(args.expt, {'rot-shift', 'rot-covariance'}))
        circular_average = sum((cos(pi*marginal_s_values(:)) + 1i*sin(pi*marginal_s_values(:))) .* s_given_x_table);
        mean_s_given_x = angle(circular_average)/pi;
        var_s_given_x = 1-abs(circular_average).^2;
    else
        mean_s_given_x = sum(marginal_s_values(:) .* s_given_x_table);
        var_s_given_x = sum((marginal_s_values(:) - mean_s_given_x).^2 .* s_given_x_table);
    end
    figure();
    subplot(1,2,1);
    contourf(xx,yy,reshape(mean_s_given_x,size(xx)),20);
    axis square; colorbar();
    title('mean of s|x');
    subplot(1,2,2);
    contourf(xx,yy,reshape(var_s_given_x,size(xx)),20);
    axis square; colorbar();
    title('variance of s|x');
end

%% (Debugging) mutual information and calibration plots
if args.debug_plot
    for i=length(priors_history):-1:1
        log_pri = log(priors_history{i});
        prior_entropy(i) = discrete_entropy(log_pri);
        s_values = linspace(-3, 3, args.variance_samples);
        avg_post = zeros(size(xx));
        parfor j=1:args.variance_samples
            rand_log_post = log_pri + logmvnpdf(xx, yy, rand_mean_fn(s_values(j), 1), rand_cov_fn(s_values(j), 1));
            avg_post = avg_post + log2prob(rand_log_post);
            rand_posterior_entropies(j) = discrete_entropy(rand_log_post);
            noiseless_log_post = log_pri + logmvnpdf(xx, yy, rand_mean_fn(s_values(j), 0), rand_cov_fn(s_values(j), 0));
            noiseless_posterior_entropies(j) = discrete_entropy(noiseless_log_post);
        end
        calibration(i) = tvd(avg_post, priors_history{i});
        avg_posterior_entropy(i,1)  = mean(rand_posterior_entropies);
        mcse_posterior_entropy(i,1) = std(rand_posterior_entropies) / sqrt(args.variance_samples);
        avg_posterior_entropy(i,2)  = mean(noiseless_posterior_entropies);
        mcse_posterior_entropy(i,2) = std(noiseless_posterior_entropies) / sqrt(args.variance_samples);
    end
    mutual_information = prior_entropy(:) - avg_posterior_entropy;
    figure; hold on;
    plot(itr_history, calibration, 'marker', '.', 'displayname', 'TVD(avg post||pri)');
    errorbar(itr_history, mutual_information(:,1), mcse_posterior_entropy(:,1), 'marker', '.', 'displayname', 'MI (w/ noise)');
    errorbar(itr_history, mutual_information(:,2), mcse_posterior_entropy(:,2), 'marker', '.', 'displayname', 'MI (noiseless)');
    xlabel('iteration');
    legend();
end

%% (Debugging) PCA plots
if args.debug_plot
    n_pc = 6;
    for i=1:length(priors_history)
        if isempty(mass_covariance{i}), continue; end
        [~, s, v] = svd(mass_covariance{i});
        s = diag(s);
        frac_s = cumsum(s)/total_variance(i);
        max_idx = max(n_pc, find(frac_s > .99, 1));
        
        fig = figure;
        subplot(1, max_idx+1, 1); hold on;
        plot(s(1:max_idx)/total_variance(i), 'marker', '.');
        alignments = abs(v' * dp_ds{i}(:)) / norm(dp_ds{i}(:));
        plot(alignments(1:max_idx), 'marker', '.');
        xlabel('rank');
        legend({'\lambda_i/total var', 'dot(e_i,dp/ds)'}, 'location', 'best');
        xlim([1 max_idx]);
        axis square;
        
        for j=1:max_idx
            subplot(1, max_idx+1, j+1);
            range = max(abs(v(:,j)));
            imagesc('XData', xvalues, 'YData', xvalues, 'CData', reshape(v(:,j), size(xx)), [-range, range]);
            title({sprintf('%%var=%.3f', s(j)/total_variance(i)), sprintf('align=%.3f', alignments(j))});
            set(gca, 'YDir', 'normal');
            axis image;
            colormap(redgreen);
            xticks([]); yticks([]);
        end
        fig.PaperUnits='inches';
        fig.PaperSize=[12,3];
        fig.PaperPosition=[0,0,12,3];
        saveas(fig, sprintf('Fig4_PCA_%s_%03d.fig', lower(args.expt), itr_history(i)));
        saveas(fig, sprintf('Fig4_PCA_%s_%03d.png', lower(args.expt), itr_history(i)));
        close(fig);
    end
end

%% Paper plot
fig = figure;
% green --> orange fade
s_colors = [linspace(0, 1, args.n_plot_s); linspace(.8, .64, args.n_plot_s); zeros(1, args.n_plot_s)]';
s_values = linspace(-3, 3, args.n_plot_s);

ax = subplot(2,3,1); hold on;
for i=1:args.n_plot_s
    noiseless_log_like = logmvnpdf(xx, yy, rand_mean_fn(s_values(i), 0), rand_cov_fn(s_values(i), 0));
    mycontour(ax, xx, yy, log2prob(noiseless_log_like), 1, '-', s_colors(i,:));
end
if strcmpi(args.expt, 'SHIFT')
    % Underlay the cubic mean function
    uistack(plot(marginal_s_values, (marginal_s_values(:)+marginal_s_values(:).^3)/10, '-k'), 'bottom');
end
xlim(ax, [min(xvalues) max(xvalues)]);
ylim(ax, [min(xvalues) max(xvalues)]);
axis square;
title('Parameterization');

ax = subplot(2,3,2); hold on;
for i=1:args.n_contours
    noisy_log_like = logmvnpdf(xx, yy, rand_mean_fn(0, 1), rand_cov_fn(0, 1));
    mycontour(ax, xx, yy, log2prob(noisy_log_like), 1, [.9 0 0], [.3 0 0], .2);
end
if strcmpi(args.expt, 'SHIFT')
    % Underlay the cubic mean function
    uistack(plot(marginal_s_values, (marginal_s_values(:)+marginal_s_values(:).^3)/10, '-k'), 'bottom');
end
xlim(ax, [min(xvalues) max(xvalues)]);
ylim(ax, [min(xvalues) max(xvalues)]);
axis square;
title('Likelihoods at s=0');

ax = subplot(2,3,3); hold on;
contourf(ax, xx, yy, prior_symmetry(log2prob(priors_history{end})), 12);
axis square;
title('Learned prior');

ax = subplot(2,3,4); hold on;
for i=1:args.n_contours
    noisy_log_like = logmvnpdf(xx, yy, rand_mean_fn(0, 1), rand_cov_fn(0, 1));
    noisy_log_post = priors_history{end} + noisy_log_like;
    mycontour(ax, xx, yy, log2prob(noisy_log_post), 1, [.8 0 .8], [.2 0 .2], .2);
end
if strcmpi(args.expt, 'SHIFT')
    % Underlay the cubic mean function
    uistack(plot(marginal_s_values, (marginal_s_values(:)+marginal_s_values(:).^3)/10, '-k'), 'bottom');
end
xlim(ax, [min(xvalues) max(xvalues)]);
ylim(ax, [min(xvalues) max(xvalues)]);
axis square;
title('Posteriors at s=0');

ax = subplot(2,3,5); hold on;
contourf(ax, xx, yy, dp_ds_symmetry(dp_ds{end}), 12);
axis square;
title('Derivative dp/ds');

ax = subplot(2,3,6); hold on;
switch lower(args.varplot)
    case 'lines'
        yyaxis left;
        plot(ax, itr_history, var_along_dpds, 'displayname', 'var along dp/ds');
        plot(ax, itr_history, total_variance, 'displayname', 'total var');
        ylim([0 inf]);
        ylabel('total variance');
        yyaxis right;
        plot(ax, itr_history, var_along_dpds ./ total_variance, 'displayname', 'frac var along dp/ds');
        ylim([0 inf]);
        ylabel({'fraction of variance', 'in dp/ds direction'});
        xlabel('learning iteration');
        legend('location', 'best')
    case 'bar'
        bar(var_along_dpds([1 end]) ./ total_variance([1 end]));
        set(gca, 'xtick', [1 2], 'xticklabel', {'before learning', 'after learning'});
        title('frac. var. along dp/ds');
end
end

%% Helpers

function p_s_given_x = likelihood(x, y, lh_fn_mu, lh_fn_cov, s_values, n_noise)
p_s_given_x = zeros(numel(s_values), numel(x));
for j=1:n_noise
    tmp_table = zeros(size(p_s_given_x));
    parfor i=1:numel(s_values)
        this_unnorm_loglike = logmvnpdf(x, y, lh_fn_mu(s_values(i),1), lh_fn_cov(s_values(i),1));
        tmp_table(i,:) = exp(this_unnorm_loglike(:));
    end
    % Normalize across s dimension for this set of noise draws, separately for each xy value
    tmp_table = tmp_table ./ sum(tmp_table, 1);
    % Average over noise draws in the full s|x table
    p_s_given_x = p_s_given_x + tmp_table / n_noise;
end
end

function avg_dp_ds = estimate_dp_ds(xx, yy,  lh_fn_mu, lh_fn_cov, prior, n_samples)
avg_dp_ds = zeros(size(xx));
ds = .01;
log_prior = log(prior);
for i=1:n_samples
    % Evaluate +∆s and -∆s with the same seed to reduce variance
    state = rng();
    log_post_pos = log_prior + logmvnpdf(xx, yy, lh_fn_mu(+ds, 1), lh_fn_cov(+ds, 1));
    rng(state);
    log_post_neg = log_prior + logmvnpdf(xx, yy, lh_fn_mu(-ds, 1), lh_fn_cov(-ds, 1));
    dp_ds = (log2prob(log_post_pos) - log2prob(log_post_neg)) / (2*ds);
    avg_dp_ds = avg_dp_ds + dp_ds / n_samples;
end
end

function [error, new_prior] = learning_step(p_s_given_x, prior_x, true_marginal_s, lr)
marginal_s = p_s_given_x * prior_x(:);
error = tvd(marginal_s, true_marginal_s);
% Chain rule to get dError/d
derror_dmarg = marginal_s(:) - true_marginal_s(:);
derror_dprior = derror_dmarg(:)' * p_s_given_x;
% Take a gradient descent step
new_prior = max(prior_x - lr * reshape(derror_dprior, size(prior_x)), 0);
new_prior = new_prior / sum(new_prior(:));
end

function p = log2prob(logp)
p = exp(logp(:) - max(logp(:)));
p = reshape(p / sum(p), size(logp));
end

function log_probs = logmvnpdf(xx, yy, mu, sigma)
%Evaluate the 2D log probability of (x,y) for the 2D gaussian with mean mu and covariance sigma...
%but clip values beyond 3 sigma to avoid visual artifacts of light-tailedness in a discrete space
xy = [xx(:) yy(:)];
R = cholcov(sigma);
z = (xy-mu)/R;
logSqrtDetSigma = sum(log(diag(R)));
log_probs = -1/2*sum(z.*z, 2) - logSqrtDetSigma;
log_probs = reshape(log_probs, size(xx));
% Enforce a minimum value equal to whatever value we get above for z=[3,0] or ||z||=3*3, since this
% will be the 3-sigma value
min_value = -1/2*(3*3)-logSqrtDetSigma;
log_probs = max(log_probs, min_value);
end

function C = randcov(k)
%Create a random covariance matrix by adding variance in k random directions
L = randn(2,k);
C = L*L';
end

function d = tvd(p, q)
% Total variational distance
d = sum(abs(p(:)/sum(p(:)) - q(:)/sum(q(:))));
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

function lse = logsumexp(vals)
log_offset = max(vals(:));
sumexp = sum(exp(vals(:) - log_offset));
lse = log(sumexp(:)) + log_offset;
end

function h = discrete_entropy(logp)
logp = logp - logsumexp(logp);
plogp = -exp(logp(:)).*logp(:);
plogp(isnan(plogp) | isinf(plogp) | plogp<0) = 0;
h = sum(plogp);
end

function r = rot(angle)
c = cos(angle); s = sin(angle);
r = [c -s; s c];
end
