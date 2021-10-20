sd = 472413084;  % random seed chosen by keyboard-mashing
iterations = 10;

% Learning for first row of fig 4
var_lh_learn('shift', sd, iterations);
% Learning for second row of fig 4
var_lh_learn('covariance', sd, iterations);

% Plotting for first row of fig 4
var_lh_plot('shift', 0, iterations);
% Plotting for second row of fig 4
var_lh_plot('covariance', 0, iterations);