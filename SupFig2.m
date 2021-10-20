function frames = mog_derivs

prior_C = .5;
var_s_given_x = .5;

xs = linspace(-4, 4, 1000);

% Case 1: constant variance s|C but shrinking means
mu_s_values = linspace(2,1e-6,6);
var_s_values = .3 * ones(1,6);

% % Case 2: constant means but shrinking variance s|C
% mu_s_values = .5 * ones(1, 6);
% var_s_values = logspace(0, -5, 6);

snapshots = 1:6; %round(linspace(1, length(mu_s_values), 5));
snapcolors = flipud(parula(length(snapshots)));

fig1 = figure;
% fig2 = figure;

for ival=1:length(mu_s_values)
    % Model learns 'true' mean since everything is gaussian
    mu_x = mu_s_values(ival);
    % Variance of stimuli themselves
    var_s_given_C = var_s_values(ival);
    
    % To satisfy circularity conditions, we have (sort of) two equations and two unknowns:
    %   1. (posterior precision on x) = likelihood precision + (prior precision)
    %   2. (prior variance on x) = var(s|C) + (posterior variance)
    % This is approximate since the prior won't be exactly gaussian. These imply...
    prior_var_x = (var_s_given_C + sqrt(var_s_given_C*(1+4*var_s_given_C*var_s_given_x)))/2;
    post_var_x_given_s = (1/prior_var_x + 1/var_s_given_x)^-1;

    % Prior on x is E_s[p(x|s)], so variances sum. 
    belief_prior_x = @(pi_c) mog.create([+mu_x -mu_x], sqrt([prior_var_x prior_var_x]), [pi_c 1-pi_c]);
    % True prior is 'belief prior' evaluated at pi=prior_C
    true_prior_x = belief_prior_x(prior_C);
    % Given an s, the likelihood has variance 'var_s_given_x'
    likelihood_s_x = @(s) mog.create(s, sqrt(var_s_given_x), 1);
    
    delta_s = .01;
    dp_ds = mog.pdf(xs, true_prior_x) .* (mog.pdf(xs, likelihood_s_x(delta_s)) - mog.pdf(xs, likelihood_s_x(-delta_s))) / (2*delta_s);
    dp_ds = dp_ds / max(dp_ds);
    
    delta_pi = .01;
    ppi = @(pi) mog.prod(likelihood_s_x(0), belief_prior_x(pi));
    dp_dpi = (mog.pdf(xs, ppi(.5+delta_pi)) - mog.pdf(xs, ppi(.5-delta_pi))) / (2*delta_pi);
    dp_dpi = dp_dpi / max(dp_dpi);
    
    %% Fig 1 : overlay
    figure(fig1);
    subplot(1,3,1); hold on;
    plot(xs, mog.pdf(xs, true_prior_x), 'Color', snapcolors(ival == snapshots, :));
    subplot(2,3,2); hold on;
    plot(xs, dp_ds, 'Color', snapcolors(ival == snapshots, :));
    subplot(2,3,5); hold on;
    plot(xs, dp_dpi, 'Color', snapcolors(ival == snapshots, :));
    subplot(2,3,3); hold on;
    plot(xs, dp_ds - dp_dpi, 'Color', snapcolors(ival == snapshots, :));
    subplot(2,3,6); hold on;
    plot(xs, dp_ds ./ dp_dpi, 'Color', snapcolors(ival == snapshots, :));
    
    %% Fig 2 : movie
%     figure(fig2); clf;
%     
%     % Distributions
%     subplot(1,2,1);
%     plot(xs, mog.pdf(xs, true_prior_x), '-k', 'LineWidth', 2);
%     axis tight;
% 
%     % Derivatives
%     subplot(1,2,2); hold on;
%     plot(xs, dp_ds, 'LineWidth', 2);
%     plot(xs, dp_dpi, 'LineWidth', 2);
%     legend('dp/ds', 'dp/d\pi', 'location', 'northeast');
%     ylim([-1, 1]);
    
    %%
    frames(ival) = getframe(gcf);
end

figure(fig1);
subplot(1,3,1); title('prior p(x)')
subplot(2,3,2); title('dp/ds');
ylim([-1.05 1.05]);
xlim([-2 2]);
subplot(2,3,5); title('dp/d\pi');
ylim([-1.05 1.05]);
xlim([-2 2]);
subplot(2,3,3); title('difference');
xlim([-2 2]);
subplot(2,3,6); title('ratio');
xlim([-2 2]);
end