function fig = Fig4(varargin)
p = inputParser;
p.addRequired('expt');  % experiment - one of 'shift' or 'covariance'
p.addParameter('eps', 5e-4); % stop learning when TVD(prior_t,prior_t-1) < eps
p.addParameter('lambda_uniform', 0.0); % weight on regularization: pull towards uniform prior
p.addParameter('lambda_entropy', 0.001); % weight on regularization: entropy of prior
p.addParameter('sigma_i', 0.2);
p.addParameter('runs', 10); % run the whole thing this many times for error bars
p.addParameter('max_iterations', 1000); % ...or stop when max is hit
p.addParameter('samples_per_iter', 5000); % number of 'data points' per iteration
p.addParameter('samples_per_derivative', 10000); % number of noise draws to estimate dp/ds
p.addParameter('variance_samples', 10000); % number of noise draws to estimate %variance
p.addParameter('rank_variance_approx', 300); % Set how many top eigenvectors to save for posterior mass covariance approximation
p.addParameter('n_plot_s', 11); % 1st plot panel: how many values of s to show
p.addParameter('n_contours', 20); % 2nd and 3rd plot panel: how many sampled likelihoods/posteriors to show
p.addParameter('analyze_every', 10); % Do dp/ds analysis on iterations (1+k*analyze_every), and final iteration
p.addParameter('lr', 1.0); % initial learning rate
p.addParameter('lr_halflife', 50); % halve learning rate every this-many iterations
p.addParameter('symmetry', true); % whether to enforce rotational symmetries of prior and dp/ds (makes it look nicer and reduces noise in plots)
p.addParameter('varplot', 'bar'); % or 'lines'
p.addParameter('savedir', 'learning');
p.addParameter('verbose', false);
p.addParameter('debug_plot', false);
p.parse(varargin{:});
args = p.Results;

xvalues = linspace(-5, 5, 85);
[xx, yy] = meshgrid(xvalues);

%% Define how observations depend on 's' - selected by args.expt
switch upper(args.expt)
    case 'SHIFT'
        % Parameterize how the mean of the likelihood depends on s (cubic)
        mean_fn = @(s) [s(:), (s(:)+s(:).^3)/10];
        % Parameterize how the covariance of the likelihood depends on s (constant)
        cov_fn = @(s) 0.5*eye(2);
        
        prior_symmetry = @(p) (p + rot90(p,2))/2;
        dp_ds_symmetry = @(dpds) (dpds - rot90(dpds,2))/2;
        
        % Create true p(s) as a uniform distribution on [-3,3]
        marginal.s_values = linspace(-3, 3, args.samples_per_iter);
        marginal.s_prob = ones(size(marginal.s_values));
        marginal.s_prob = marginal.s_prob / sum(marginal.s_prob);
    case 'COVARIANCE'
        % Parameterize how the mean of the likelihood depends on s (constant)
        mean_fn = @(s) [0 0];
        covplus = [1 1; 1 1];
        covzero = 0.5*eye(2);
        covminus = [1 -1; -1 1];
        cov_fn = @(s) covzero + (s>0)*covplus*tanh(s) + (s<0)*covminus*abs(tanh(s));
        
        prior_symmetry = @(p) (p + rot90(p,1) + rot90(p,2) + rot90(p,3))/4;
        dp_ds_symmetry = @(dpds) (dpds - rot90(dpds,1) + rot90(dpds,2) - rot90(dpds,3))/4;
        
        s = eigs(cov_fn(+3));
        fprintf('[DEBUG] Narrowest E|s stdev is %f; E|x is %f\n', sqrt(min(s)), args.sigma_i);
        
        % Create true p(s) as a uniform distribution on [-3,3]
        marginal.s_values = linspace(-3, 3, args.samples_per_iter);
        marginal.s_prob = ones(size(marginal.s_values));
        marginal.s_prob = marginal.s_prob / sum(marginal.s_prob);
    case 'ROT-SHIFT'
        % Parameterize how the mean of the likelihood depends on s (around a ring)
        mean_fn = @(s) 1.5*[cos(pi*s), sin(pi*s)];
        % Parameterize how the covariance of the likelihood depends on s (constant)
        cov_fn = @(s) 0.5*eye(2);
        
        prior_symmetry = @(p) (p + rot90(p,1) + rot90(p,2) + rot90(p,3))/4;
        dp_ds_symmetry = @(dpds) (dpds - flipud(dpds))/2;
        
        % Create true p(s) uniform on [-1,1]
        marginal.s_values = linspace(-1, 1, args.samples_per_iter);
        marginal.s_prob = ones(size(marginal.s_values));
        marginal.s_prob = marginal.s_prob / sum(marginal.s_prob);
    case 'ROT-COVARIANCE'
        % Parameterize how the mean of the likelihood depends on s (constant)
        mean_fn = @(s) [0 0];
        % Parameterize how the covariance of the likelihood depends on s (elongated, pointing to s along the ring)
        cov_fn = @(s) rot(pi/2*s)'*[2 0; 0 0.25]*rot(pi/2*s) + randcov(2)*noise*cov_noise_factor;
        
        prior_symmetry = @(p) (p + rot90(p,1) + rot90(p,2) + rot90(p,3))/4;
        dp_ds_symmetry = @(dpds) (dpds - flipud(dpds))/2;
        
        % Create true p(s) uniform on [-1,1]
        marginal.s_values = linspace(-1, 1, args.samples_per_iter);
        marginal.s_prob = ones(size(marginal.s_values));
        marginal.s_prob = marginal.s_prob / sum(marginal.s_prob);
    otherwise
        error('Need EXPERIMENT to be ''SHIFT'' or ''COVARIANCE''.');
end

% compute true ('world') marginal distribution over E
marginal.evidence = zeros(size(xx));
for i=1:length(marginal.s_values)
    s = marginal.s_values(i);
    w = marginal.s_prob(i);
    prob_ev_given_s = exp(logmvnpdf(xx, yy, mean_fn(s), cov_fn(s)));
    marginal.evidence = marginal.evidence + w * prob_ev_given_s;
end
marginal.evidence = marginal.evidence / sum(marginal.evidence(:));

% cov_ix is the covariance of I|x in the generative model.
cov_ix = args.sigma_i.^2*eye(2);
% Rand log likelihood is a function handle that takes in a value of s, produces a random I|s, and
% returns the likelihood 'density' I|x for all x
rand_log_likelihood = @(s,nz) logmvnpdf(xx, yy, mvnrnd(mean_fn(s), cov_fn(s)), cov_ix);

% Visualize this random-log-likelihood function
if args.debug_plot
    figure;
    ax = gca;
    n_vis = 50;
    % green --> orange fade for s=-1 --> +1
    s_colors = [linspace(0, 1, n_vis); linspace(.8, .64, n_vis); zeros(1, n_vis)]';
    idx = round(linspace(1, args.samples_per_iter, n_vis+2));
    s_values = marginal.s_values(idx(2:end-1));
    hold on;
    for i=1:n_vis
        log_like = rand_log_likelihood(s_values(i), 0);
        mycontour(ax, xx, yy, log2prob(log_like), 1, '-', s_colors(i,:));
    end
    if strcmpi(args.expt, 'SHIFT')
        % Underlay the cubic mean function
        uistack(plot(marginal.s_values, (marginal.s_values(:)+marginal.s_values(:).^3)/10, '-k'), 'bottom');
    end
    xlim(ax, [min(xvalues) max(xvalues)]);
    ylim(ax, [min(xvalues) max(xvalues)]);
    axis square;
    title('Parameterization w/out noise');
end

%% Run analysis or load from file if available
for r=1:args.runs
    save_file = fullfile(args.savedir, sprintf('Fig4_%s_%.3f_%.3f_%.2f_%02d.mat', lower(args.expt), args.lambda_uniform, args.lambda_entropy, args.sigma_i, r));
    if exist(save_file, 'file')
        data = load(save_file);
        log_priors_history = data.log_priors_history;
        itr_history = data.itr_history;
        dp_ds = data.dp_ds;
        total_variance = data.total_variance;
        var_along_dpds = data.var_along_dpds;
        mass_covariance_lowrank = data.mass_covariance_lowrank;
        clear data;
        if args.verbose
            fprintf('Loaded precomputed results from %s with %d learning iterations\n', save_file, max(itr_history));
        end
    else
        %% Iterate learning by moving the prior such that the marginal likelihood gets closer to the true p(s)
        % Initialize prior on x to uniform
        log_prior = -ones(size(xx)) * numel(xx);
        log_priors_history = {};
        itr_history = [];
        delta = [nan];
        delta0 = [nan];
        for itr=1:args.max_iterations+1
            if mod(itr-1, args.analyze_every) == 0
                log_priors_history{end+1} = log_prior;
                itr_history(end+1) = itr-1;
            end

            lr = args.lr / 2.^floor(itr/args.lr_halflife);
            log_prior = learning_step(log_prior, rand_log_likelihood, marginal, args.lambda_uniform, args.lambda_entropy, lr);

            if args.symmetry
                log_prior = log(prior_symmetry(exp(log_prior)));
                log_prior = log_prior - logsumexp(log_prior);
            end

            if itr >= 2
                delta(itr) = tvd(last_log_prior, log_prior, true);
                delta0(itr) = tvd(log_priors_history{1}, log_prior, true);
                if args.verbose
                    fprintf('Learning itr %03d\tTVD change in prior = %f\tTVD from init = %f\n', itr-1, delta(itr), delta0(itr));
                end

                if delta(itr) < args.eps || delta0(end) < delta0(end-1)
                    if args.verbose
                        fprintf('Breaking after %d learning iterations\n', itr-1);
                    end
                    log_priors_history{end+1} = log_prior;
                    itr_history(end+1) = itr;
                    break
                end
            end
            last_log_prior = log_prior;
        end

        if args.debug_plot
            figure;
            for i=1:length(log_priors_history)
                subplotsquare(length(log_priors_history), i);
                contourf(xx, yy, exp(log_priors_history{i}), linspace(0, exp(max(log_priors_history{i}(:))), 21));
                axis square;
                xticks([]); yticks([]);
            end
        end
        if args.verbose && itr >= args.max_iterations
            fprintf('Failed to converge at eps=%.5f after %d iterations\n', args.eps, args.max_iterations);
        end

        %% Estimate E_{noise}[dp/ds] at s=0, once for each learning iteration
        for i=length(log_priors_history):-1:1
            dp_ds{i} = estimate_dp_ds(rand_log_likelihood, log_priors_history{i}, args.samples_per_derivative);

            if args.symmetry
                dp_ds{i} = dp_ds_symmetry(dp_ds{i});
            end

            if args.verbose
                fprintf('dp/ds itr %03d\n', itr_history(i));
            end

            if args.debug_plot
                subplotsquare(length(log_priors_history), i);
                contourf(xx, yy, dp_ds{i}, 20);
                axis square;
                xticks([]); yticks([]);
                drawnow;
            end
        end

        %% Estimate total variance and variance in dp/ds direction after each iteration
        for i=length(log_priors_history):-1:1
            posterior_instances = zeros(args.variance_samples, numel(xx));
            log_pri = log_priors_history{i};
            parfor j=1:args.variance_samples
                % New random posterior at s=0 using prior at iteration itr.
                rand_log_post = log_pri + rand_log_likelihood(0, 1);
                posterior_instances(j,:) = log2prob(rand_log_post(:));
            end

            % Get total variance and projected variance
            mass_covariance = cov(posterior_instances);
            % since mass_covariance is a huge matrix, we'll store low-rank R such that mass_covariance
            % is approximately R*R'.
            [~,s,v] = svds(mass_covariance, args.rank_variance_approx);
            mass_covariance_lowrank{i} = sqrt(diag(s))'.*v;
            total_variance(i) = trace(mass_covariance);
            norm_dp_ds = dp_ds{i}(:) / sqrt(sum(dp_ds{i}(:).^2));
            var_along_dpds(i) = var(posterior_instances * norm_dp_ds);

            if args.verbose
                fprintf('variance itr %03d\tvar = %f\tdp/ds var = %f\tfrac var = %f\n', itr_history(i), total_variance(i), var_along_dpds(i), var_along_dpds(i)/total_variance(i));
            end
        end

        %% Cache all workspace variables to a file
        clear posterior_instances mass_covariance s v;
        save(save_file);
    end
end

%% (Debugging) mutual information, log marginal, and calibration plots
if args.debug_plot
    for i=length(log_priors_history):-1:1
        if args.verbose, fprintf('Computing model stats itr %d\n', itr_history(i)); end
        log_pri = log_priors_history{i};
        pri = exp(log_priors_history{i} - logsumexp(log_priors_history{i}));
        prior_entropy(i) = discrete_entropy(log_pri);
        s_values = linspace(-3, 3, args.variance_samples);
        avg_post = zeros(size(xx));
        gen_marginal_likelihood = zeros(size(marginal.evidence));
        for j=1:numel(xx)
            prob_ev_given_x = exp(logmvnpdf(xx, yy, [xx(j) yy(j)], cov_ix));
            gen_marginal_likelihood = gen_marginal_likelihood + pri(j) * prob_ev_given_x;
        end
        gen_marginal_likelihood = gen_marginal_likelihood / sum(gen_marginal_likelihood(:));
        parfor j=1:args.variance_samples
            rand_log_post = log_pri + rand_log_likelihood(s_values(j));
            avg_post = avg_post + log2prob(rand_log_post);
            rand_posterior_entropies(j) = discrete_entropy(rand_log_post);
        end
        data_fit(i) = tvd(marginal.evidence, gen_marginal_likelihood, false);
        calibration(i) = tvd(avg_post, pri, false);
        avg_posterior_entropy(i)  = mean(rand_posterior_entropies);
        mcse_posterior_entropy(i) = std(rand_posterior_entropies) / sqrt(args.variance_samples);
    end
    mutual_information = prior_entropy - avg_posterior_entropy;
    figure; hold on;
    plot(itr_history, calibration, 'marker', '.', 'displayname', 'TVD(avg post||pri)');
    errorbar(itr_history, mutual_information, mcse_posterior_entropy, 'marker', '.', 'displayname', 'MI');
    plot(itr_history, data_fit, 'marker', '.', 'displayname', 'TVD(p(I)||q(I))');
    xlabel('iteration');
    legend();
end

%% Paper plot
fig = figure;
% green --> orange fade
s_colors = [linspace(0, 1, args.n_plot_s); linspace(.8, .64, args.n_plot_s); zeros(1, args.n_plot_s)]';
s_values = linspace(-3, 3, args.n_plot_s);

ax = subplot(2,3,1); hold on;
for i=1:args.n_plot_s
    noiseless_log_like = logmvnpdf(xx, yy, mean_fn(s_values(i)), cov_fn(s_values(i)));
    mycontour(ax, xx, yy, log2prob(noiseless_log_like), 1, '-', s_colors(i,:));
end
if strcmpi(args.expt, 'SHIFT')
    % Underlay the cubic mean function
    uistack(plot(marginal.s_values, (marginal.s_values(:)+marginal.s_values(:).^3)/10, '-k'), 'bottom');
end
xlim(ax, [min(xvalues) max(xvalues)]);
ylim(ax, [min(xvalues) max(xvalues)]);
axis square;
title('Parameterization');

ax = subplot(2,3,2); hold on;
state = rng();
for i=1:args.n_contours
    noisy_log_like = rand_log_likelihood(0);
    mycontour(ax, xx, yy, log2prob(noisy_log_like), 1, [.9 0 0], [.3 0 0], .2);
end
if strcmpi(args.expt, 'SHIFT')
    % Underlay the cubic mean function
    uistack(plot(marginal.s_values, (marginal.s_values(:)+marginal.s_values(:).^3)/10, '-k'), 'bottom');
end
xlim(ax, [min(xvalues) max(xvalues)]);
ylim(ax, [min(xvalues) max(xvalues)]);
axis square;
title('Likelihoods at s=0');

ax = subplot(2,3,3); hold on;
contourf(ax, xx, yy, prior_symmetry(log2prob(log_priors_history{end})), 12);
axis square;
title('Learned prior');

ax = subplot(2,3,4); hold on;
rng(state); % same RNG state for posteriors and likelihoods
for i=1:args.n_contours
    noisy_log_like = rand_log_likelihood(0);
    noisy_log_post = log_priors_history{end} + noisy_log_like;
    mycontour(ax, xx, yy, log2prob(noisy_log_post), 1, [.8 0 .8], [.2 0 .2], .2);
end
if strcmpi(args.expt, 'SHIFT')
    % Underlay the cubic mean function
    uistack(plot(marginal.s_values, (marginal.s_values(:)+marginal.s_values(:).^3)/10, '-k'), 'bottom');
end
xlim(ax, [min(xvalues) max(xvalues)]);
ylim(ax, [min(xvalues) max(xvalues)]);
axis square;
title('Posteriors at s=0');

ax = subplot(2,3,5); hold on;
contourf(ax, xx, yy, dp_ds_symmetry(dp_ds{end}), 12);
axis square;
title('Derivative dp/ds');

axis_colors = lines(2);
    
    function onoff = legendhandle(r)
        if r==1, onoff = 'on';
        else, onoff = 'off';
        end
    end

ax = subplot(2,3,6); hold on;
switch lower(args.varplot)
    case 'lines'
        yyaxis left;
        for r=1:args.runs
            save_file = fullfile(args.savedir, sprintf('Fig4_%s_%.3f_%.3f_%.2f_%02d.mat', ...
                lower(args.expt), args.lambda_uniform, args.lambda_entropy, args.sigma_i, r));
            data = load(save_file, 'total_variance', 'var_along_dpds', 'itr_history');
            plot(ax, data.itr_history, data.var_along_dpds, 'displayname', 'var along dp/ds', ...
                'color', axis_colors(1,:), 'linestyle', '-', 'handlevisibility', legendhandle(r));
            plot(ax, data.itr_history, data.total_variance, 'displayname', 'total var', ...
                'color', axis_colors(1,:), 'linestyle', '--', 'handlevisibility', legendhandle(r));
        end
        ylim([0 inf]);
        ylabel('total variance');
        yyaxis right;
        for r=1:args.runs
            save_file = fullfile(args.savedir, sprintf('Fig4_%s_%.3f_%.3f_%.2f_%02d.mat', ...
                lower(args.expt), args.lambda_uniform, args.lambda_entropy, args.sigma_i, r));
            data = load(save_file, 'total_variance', 'var_along_dpds', 'itr_history');
            plot(ax, data.itr_history, data.var_along_dpds ./ data.total_variance, 'displayname', 'frac var along dp/ds', ...
                'color', axis_colors(2,:), 'linestyle', '-', 'handlevisibility', legendhandle(r));
        end
        ylim([0 inf]);
        ylabel({'fraction of variance', 'in dp/ds direction'});
        xlabel('learning iteration');
        legend('location', 'best')
    case 'bar'
        for r=args.runs:-1:1
            save_file = fullfile(args.savedir, sprintf('Fig4_%s_%.3f_%.3f_%.2f_%02d.mat', ...
                lower(args.expt), args.lambda_uniform, args.lambda_entropy, args.sigma_i, r));
            data = load(save_file, 'total_variance', 'var_along_dpds', 'itr_history');
            all_frac_var_along(:,r) = data.var_along_dpds ./ data.total_variance;
        end
        mean_frac = mean(all_frac_var_along([1 end], :), 2);
        sem_frac = std(all_frac_var_along([1 end], :), [], 2) / sqrt(args.runs);
        bar([1 2], mean_frac);
        errorbar([1 2], mean_frac, sem_frac, 'k', 'linestyle', 'none', 'marker', '.');
        set(gca, 'xtick', [1 2], 'xticklabel', {'before learning', 'after learning'});
        title('frac. var. along dp/ds');
end
end

%% Helpers

function avg_dp_ds = estimate_dp_ds(rand_ll_fn, log_prior, n_samples)
avg_dp_ds = zeros(size(log_prior));
ds = .01;
for i=1:n_samples
    % Evaluate +∆s and -∆s with the same seed to reduce variance
    state = rng();
    log_post_pos = log_prior + rand_ll_fn(+ds, 1);
    rng(state);
    log_post_neg = log_prior + rand_ll_fn(-ds, 1);
    dp_ds = (log2prob(log_post_pos) - log2prob(log_post_neg)) / (2*ds);
    avg_dp_ds = avg_dp_ds + dp_ds / n_samples;
end
end

function new_log_prior = learning_step(log_prior_x, rand_ll, marginal, lambda_unif, lambda_ent, lr)
uniform = ones(size(log_prior_x)) / numel(log_prior_x);
prior_x = log2prob(log_prior_x);
avg_post_x = zeros(size(log_prior_x));
for i=1:length(marginal.s_values)
    s = marginal.s_values(i);
    rand_log_post = log_prior_x + rand_ll(s, 1);
    rand_post = exp(rand_log_post - logsumexp(rand_log_post));
    avg_post_x = avg_post_x + marginal.s_prob(i) * rand_post;
end
avg_post_x = avg_post_x / sum(avg_post_x(:));

% Gradient of KL(p(evidence)||q(evidence)) with respect to log(q(x)) is:
grad_kl_ev = -avg_post_x;
% Gradient of KL(u(x)||q(x)) with respect to log(q(x)) is:
grad_kl_unif = -uniform;
% Gradient of H[q] with respect to log(q(x)) is:
grad_entropy = -prior_x .* (1 + log_prior_x);

% % === DEBUG GRADIENT CHECKS ===
% eps = 1e-5;
% gen_marginal_likelihood = zeros(size(marginal.evidence));
% [xx,yy] = meshgrid(linspace(-5,5,85),linspace(-5,5,85)); cov_ix = .04*eye(2);
% for j=1:numel(xx)
%     prob_ev_given_x = exp(logmvnpdf(xx, yy, [xx(j) yy(j)], cov_ix));
%     gen_marginal_likelihood = gen_marginal_likelihood + prior_x(j) * prob_ev_given_x;
% end
% gen_marginal_likelihood = gen_marginal_likelihood / sum(gen_marginal_likelihood(:));
% kl_ev_baseline = discrete_kl(log(marginal.evidence), log(gen_marginal_likelihood));
% num_grad_kl_ev = zeros(size(grad_kl_ev));
% for j=1:numel(xx)
%     prob_ev_given_x = exp(logmvnpdf(xx, yy, [xx(j) yy(j)], cov_ix));
%     new_gen_marginal_likelihood = (1-eps)*gen_marginal_likelihood + eps*prob_ev_given_x/sum(prob_ev_given_x(:));
%     new_kl_ev = discrete_kl(log(marginal.evidence), log(new_gen_marginal_likelihood));
%     num_grad_kl_ev(j) = (new_kl_ev - kl_ev_baseline) / eps;
%     num_grad_kl_ev(j) = prior_x(j) * num_grad_kl_ev(j); % change of variables from grad wrt prob to grad wrt log prob
% end
% projected_grad_kl_ev = grad_kl_ev - prior_x * sum(grad_kl_ev(:)'*prior_x(:))/sum(prior_x(:)'*prior_x(:));
% subplot(1,3,1); hold on;
% scatter(num_grad_kl_ev(:), grad_kl_ev(:), '.');
% scatter(num_grad_kl_ev(:), projected_grad_kl_ev(:), '.');
% axis square; axis equal; grid on;
% plot(xlim, xlim, '-k');
% 
% num_grad_kl_unif = zeros(size(grad_kl_unif));
% kl_unif_baseline = discrete_kl(log(uniform), log_prior_x);
% for j=1:numel(xx)
%     delta = zeros(size(xx));
%     delta(j) = 1;
%     new_kl_unif = discrete_kl(log(uniform), log(prior_x*(1-eps) + eps*delta));
%     num_grad_kl_unif(j) = (new_kl_unif - kl_unif_baseline) / eps;
%     num_grad_kl_unif(j) = prior_x(j) * num_grad_kl_unif(j); % change of variables from grad wrt prob to grad wrt log prob
% end
% projected_grad_kl_unif = grad_kl_unif - prior_x * sum(grad_kl_unif(:)'*prior_x(:))/sum(prior_x(:)'*prior_x(:));
% subplot(1,3,2); hold on;
% scatter(num_grad_kl_unif(:), grad_kl_unif(:), '.');
% scatter(num_grad_kl_unif(:), projected_grad_kl_unif(:), '.');
% axis square; axis equal; grid on;
% plot(xlim, xlim, '-k');
% 
% num_grad_entropy = zeros(size(grad_entropy));
% ent_baseline = discrete_entropy(log_prior_x);
% for j=1:numel(xx)
%     delta = zeros(size(xx));
%     delta(j) = 1;
%     new_ent = discrete_entropy(log(prior_x*(1-eps) + eps*delta));
%     num_grad_entropy(j) = (new_ent - ent_baseline) / eps;
%     num_grad_entropy(j) = prior_x(j) * num_grad_entropy(j); % change of variables from grad wrt prob to grad wrt log prob
% end
% projected_grad_entropy = grad_entropy - prior_x * sum(grad_entropy(:)'*prior_x(:))/sum(prior_x(:)'*prior_x(:));
% subplot(1,3,3); hold on;
% scatter(num_grad_entropy(:), grad_entropy(:), '.');
% scatter(num_grad_entropy(:), projected_grad_entropy(:), '.');
% axis square; axis equal; grid on;
% plot(xlim, xlim, '-k');
% % ============================

% Gradient of combined loss, where loss is KL(p(data)||q(data)) + lambda_u*KL(u(x)||q(x)) - lambda_e*H[q]
grad_loss = grad_kl_ev + lambda_unif * grad_kl_unif - lambda_ent * grad_entropy;
% Gradient of mass, sum(q(:)), with respect to log(q(x)) is just q(x) itself:
grad_mass = prior_x;
% Projected gradient, removing component in direction of change-in-mass
proj_grad_loss = grad_loss - grad_mass * sum(grad_loss(:)'*grad_mass(:)) / sum(grad_mass(:)'*grad_mass(:));
% Take a step and normalize again
new_log_prior = log_prior_x - lr * proj_grad_loss;
new_log_prior = new_log_prior - logsumexp(new_log_prior);
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
end

function d = tvd(p, q, islog)
% Total variational distance
if islog
    p = exp(p(:) - logsumexp(p(:)));
    q = exp(q(:) - logsumexp(q(:)));
end
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

function kl = discrete_kl(logp, logq)
p = log2prob(logp);
kl = sum(p(:).*(logp(:)-logq(:)));
end

function r = rot(angle)
c = cos(angle); s = sin(angle);
r = [c -s; s c];
end
