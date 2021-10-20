function [x_hat, y_hat] = york_map_xy(x_data, y_data, x_err, y_err, a_fit, b_fit)
% Helper function to recover 'true' x and y
%
% written by Richard Lange July 2021.

x_hat = (y_err.^2 .* x_data + x_err.^2 .* (y_data - a_fit) * b_fit) ./ (y_err.^2 + x_err.^2 * b_fit);
y_hat = x_hat * b_fit + a_fit;

end