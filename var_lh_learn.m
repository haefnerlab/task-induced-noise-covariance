function var_lh_learn(expt, sd, iters, samplesper, reguniform, sym)
% Iterate learning for SHIFT or COVARIANCE experiment. Save out learned prior for plotting by
% @var_lh_plot

% Apply random seed if given, otherwise create and record the seed
if ~exist('sd', 'var') || isempty(sd), sd = randi(2^32); end
if ~exist('learning', 'file'), mkdir('learning'); end

rng(sd, 'twister');

%% Default args

% number of learning steps [avg posterior --> prior]
if ~exist('iters', 'var') || isempty(iters), iters = 10;    end
% number of data points per learning step (num samples of s)
if ~exist('samplesper', 'var') || isempty(samplesper), samplesper = 10000; end
% Amount of uniform density added after each iteration for regularization/stability
if ~exist('reguniform', 'var') || isempty(reguniform), reguniform = .01; end
% By default, apply known symmetries to the learned prior after each iteration
if ~exist('sym', 'var') || isempty(sym), sym = true; end

%% Setup 2D space, LH functions, etc
xs = linspace(-3, 3, 51);
[xx, yy] = meshgrid(xs);

% Start with a uniform prior
init_log_prior = -log(numel(xs)) * ones(size(xx));

switch upper(expt)
    case 'SHIFT'
        % Parameterize how the mean of the likelihood depends on s
        mu_x_s = @(s) [s(:) (s(:)+s(:).^3)/10];
        
        % Parameterize how the covariance of the likelihood depends on s (it doesn't)
        cov_x_s = @(s) eye(2);
    case 'COVARIANCE'
        % Parameterize how the mean of the likelihood depends on s (it doesnt)
        mu_x_s = @(s) [0 0];
        
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

%% Iterate learning

log_prior = init_log_prior;

savename = sprintf('svres-%s-sper=%d-sd=%d', lower(expt), samplesper, sd);

for itr=1:iters
    fprintf('ITER %d/%d\n', itr, iters);
    
    % Do learning (or load from file)
    thissave = fullfile('learning', [savename '-iter=' num2str(itr) '.mat']);
    if exist(thissave, 'file')
        ld = load(thissave);
        log_prior = ld.log_prior;
    else
        % Do learning, enforcing uniform 's' by using equal grid in [-3,3]. Each LH is still
        % stochastic given s.
        svalues = linspace(-3, 3, samplesper);
        log_prior = iterate_learning(xx, yy, likelihood_fn_mu, likelihood_fn_sigma, log_prior, ...
            svalues, reguniform);
        % Enforce symmetry in the prior
        if sym
            log_prior = symmetrize(log_prior, +1, expt);
        end
        % Save this isteration
        save(thissave, 'log_prior');
    end
end
end

%% Helpers

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

function [new_log_prior, new_prior] = iterate_learning(xx, yy, lh_fn_mu, lh_fn_sig, log_prior, svalues, reguniform)
% Map sampled 's's onto (random) likelihood parameters
lh_mu = arrayfun(@(s) lh_fn_mu(s,1), svalues, 'UniformOutput', false);
lh_sig = arrayfun(@(s) lh_fn_sig(s,1), svalues, 'UniformOutput', false);

% Reset posterior for this iteration
net_posteriors = zeros(size(log_prior));

for i=1:length(svalues)
    % Compute value of LH across grid
    lh_log_prob = logmvnpdf(xx, yy, lh_mu{i}, lh_sig{i});
    
    % Accumulate posteriors
    this_log_posterior = lh_log_prob + log_prior;
    net_posteriors = net_posteriors + exp(this_log_posterior);
    
    % Display progress
    if mod(i, 1000) == 0, fprintf('\tlearning: iter %d / %d\n', i, length(svalues)); end
end

% Learning: prior = average posterior, with some light regularization for stability reasons
uniform = ones(size(log_prior)) / numel(log_prior);
new_prior = (1-reguniform)*net_posteriors/sum(net_posteriors(:)) + reguniform*uniform;
new_log_prior = log(new_prior);
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