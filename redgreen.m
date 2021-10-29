function colors = redgreen(n)
if nargin < 1, n = 64; end

if mod(n, 2) == 1
    redpart = [linspace(1, 0, (n-1)/2); zeros(1, (n-1)/2); zeros(1, (n-1)/2)]';
    gnpart = [zeros(1, (n-1)/2); linspace(0, 1, (n-1)/2); zeros(1, (n-1)/2)]';
    colors = [redpart; 0 0 0; gnpart];
else
    redpart = [linspace(1, 2/n, n/2); zeros(1, n/2); zeros(1, n/2)]';
    gnpart = [zeros(1, n/2); linspace(2/n, 1, n/2); zeros(1, n/2)]';
    colors = [redpart; gnpart];
end
end