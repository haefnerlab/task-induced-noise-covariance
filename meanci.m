function [ mean, lower, upper, median ] = meanci( data, interval, pmf )
%MEANCI compute mean and lower/upper confidence intervals, generally
%handling NaN values gracefully
%
% [mean,lower,upper,median] = MEANCI(data, [interval]) where data has measurements
% on rows and variables across columns, returns a row vector for each of
% mean, lower, and upper. interval defaults to 0.95
%
% [mean,lower,upper,median] = MEANCI(data, interval, [pmf]) where 'pmf' is
% a boolean flag that, if set to true, indicates that 'data' will be as a
% pmf (it will have 2 columns [probs(:) values(:)])

if nargin < 2, interval = 0.95; end

if ~(interval > 0 && interval < 1)
    error('confidence interval must be in (0,1)');
end

if nargin >= 3 && pmf
    % Ensure that PMF sums to 1
    data(:, 1) = data(:, 1) / sum(data(:, 1));
    % Get CMF from PMF
    cmf = cumsum(data(:, 1));
    if min(cmf) > 1-interval
        error('Not enough low probability mass to get %f interval', 1-interval);
    end
    
    mean = sum(data(:, 1) .* data(:, 2:end), 1);
    for v=size(data,2):-1:2
        % Median is where CMF reaches 0.5
        median(v-1) = interp1(cmf, data(:, v), .5);
        % Low CI is where CMF reaches 1-interval
        lower(v-1) = interp1(cmf, data(:, v), 1-interval);
        % High CI is where CMF reaches 1-interval
        upper(v-1) = interp1(cmf, data(:, v), interval);
    end
else
    mean = nanmean(data,1);
    dsorted = sort(data,1); % note that sort() puts NaNs at the end
    indexes = 1:size(data,1);

    for var=size(data,2):-1:1
        % consider only non-nan values for this var in computing the confidence
        % interval (1 thru n, where n is #non-nan)
        n = sum(~isnan(data(:,var)));

        if n == 0
            lower(var) = NaN;
            upper(var) = NaN;
        else
            lo_idx = n * (1-interval)/2;
            hi_idx = n - lo_idx;

            % get data values at lo_idx and hi_idx, which in general won't be
            % integers, we interpolate in the data for them.
            lower(var) = interp1(indexes, dsorted(:,var), lo_idx, 'pchip');
            upper(var) = interp1(indexes, dsorted(:,var), hi_idx, 'pchip');
            median(var) = (dsorted(floor(n/2), var) + dsorted(ceil(n/2), var)) / 2;
        end
    end
end
end

