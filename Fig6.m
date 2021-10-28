function fig = Fig6(varargin)
p = inputParser;
p.addOptional('stim_type', 'static', @ischar);
p.addParameter('verbose', false);
p.addParameter('debug_fig', false);
p.parse(varargin{:});
args = p.Results;

save_file = sprintf('Haefner2016_%s.mat', lower(args.stim_type));
if ~exist(save_file, 'file')
    if ~exist('sampling_decision', 'file')
        error('Need to sync git submodule for github.com/haefnerlab/sampling_decision');
    end
    if args.verbose
        fprintf('Saved model results not found - running and saving sampling_decision model from Haefner et al (2016)\n');
    end
    
    if ~exist('S_Exp_Para', 'file'), addpath(genpath('sampling_decision')); end
    
    % Run Haefner et al (2016) model with all default parameters, except 80/20 uncertainty about
    % task type, and force 1000 trials
    model_params = S_Exp_Para('paper-2AFC-corr', ...
        'G.prior_task', [.8, .2], ...
        'S.number_repetitions', 1000, ...
        'I.stimulus_regime', args.stim_type);
    model_results = S_Experiment(model_params);
    save(save_file, '-struct', 'model_results');
    clear model_results;
end

if args.verbose
    fprintf('Loading %s\n', save_file);
end

% In the model, 'X' is the variable corresponding to V1 activity, with size (trials, neurons, time)
data = load(save_file, 'X');
rates = mean(data.X, 3);
covmatrix = cov(rates);

if args.debug_fig
    figure;
    imagesc(corrcov(covmatrix) - eye(size(rates,2))); axis image; colorbar;
end

%% Paper figure
[~,s,v] = svd(covmatrix);
s = diag(s);

n_highlight = 5;
colors = lines(n_highlight);

fig = figure;
subplot(1,2,1); hold on;
plot(s(1:20), 'o');
for i=1:n_highlight
    plot(i, s(i), 'o', 'color', colors(i,:), 'markerfacecolor', colors(i,:));
end
subplot(1,2,2); hold on;
for i=n_highlight:-1:1
    plot(v(:,i), '-', 'color', colors(i,:), 'linewidth', 2);
end
end