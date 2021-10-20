save_file = 'Haefner2016.mat';
if ~exist(save_file, 'file')
    if ~exist('sampling_decision', 'file')
        error('Need to sync git submodule for github.com/haefnerlab/sampling_decision');
    end
    warning('Saved model results not found - running and saving sampling_decision model from Haefner et al (2016)');
    
    if ~exist('S_Exp_Para', 'file'), addpath(genpath('sampling_decision')); end
    
    % Run Haefner et al (2016) model with all default parameters
    model_params = S_Exp_Para('paper-2AFC-corr', 'G.prior_task', [.8, .2]);
    model_results = S_Experiment(model_params);
    save(save_file, '-struct', 'model_results');
end

% In the model, 'X' is the variable corresponding to V1 activity, with size (trials, neurons, time)
data = load(save_file, 'X');
rates = mean(data.X, 3);

[~,s,v] = svd(cov(rates));
s = diag(s);

n_highlight = 3;
colors = lines(n_highlight);

figure;
subplot(1,2,1); hold on;
plot(s(1:20), 'o');
for i=1:n_highlight
    plot(i, s(i), 'o', 'color', colors(i,:), 'markerfacecolor', colors(i,:));
end
subplot(1,2,2); hold on;
for i=1:n_highlight
    plot(v(:,i), '-', 'color', colors(i,:), 'linewidth', 2);
end