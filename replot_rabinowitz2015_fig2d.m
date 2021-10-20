function replot_rabinowitz2015_fig2d
% Data scraped from figure 2d of [2]
%
% [2] ﻿Rabinowitz, N. C., Goris, R. L. T., Cohen, M. R., & Simoncelli, E. P. (2015). Attention
%   stabilizes the shared gain of V4 populations. ELife, 4.
%   https://doi.org/http://dx.doi.org/10.7554/eLife.08998

dprime = [-0.08941048034934496
    0.2540393013100438
    0.4725982532751093
    0.7337336244541485
    1.2403930131004368];

weight = [0.03085388994307400
    0.04440227703984818
    0.05943074003795065
    0.0732068311195446
    0.08994307400379506];

weight_lo = [0.029373814041745758
    0.043149905123339646
    0.05783681214421253
    0.07149905123339659
    0.08812144212523718];

weight_hi = [0.032561669829222
    0.04554079696394686
    0.060796963946869065
    0.07445920303605313
    0.09130929791271347];

%%
cla; hold on;
color = lines(1);
plot(dprime, weight, 'linewidth', 2, 'color', color);
plot(dprime, weight_lo, 'linewidth', 1, 'color', color);
plot(dprime, weight_hi, 'linewidth', 1, 'color', color);
xlabel('d''');
ylabel('modulator weight');

end