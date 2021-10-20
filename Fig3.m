prior_sigma = .5;
prior_ctr = mog.create([-1, +1], prior_sigma*[1, 1], [.5, .5]);
prior_neg = mog.create([-1, +1], prior_sigma*[1, 1], [.55, .45]);
prior_pos = mog.create([-1, +1], prior_sigma*[1, 1], [.45, .55]);

like_ctr = mog.create(0, .3, 1);
like_neg = mog.create(-.05, .3, 1);
like_pos = mog.create(+.05, .3, 1);

xs = linspace(-4,4,500);

color_ctr = [0 0 0];
color_neg = [0 .7 0];
color_pos = [.9 .2 0];

figure;
subplot(3,3,1);
plot(xs, mog.pdf(xs, prior_ctr), '-', 'color', color_ctr);
subplot(3,3,2);
plot(xs, mog.pdf(xs, prior_ctr), '-', 'color', color_ctr);
subplot(3,3,3); hold on;
plot(xs, mog.pdf(xs, prior_neg), '-', 'color', color_neg);
plot(xs, mog.pdf(xs, prior_pos), '-', 'color', color_pos);
subplot(3,3,4);
plot(xs, mog.pdf(xs, mog.prod(prior_ctr, like_ctr)), '-', 'color', color_ctr);
subplot(3,3,5); hold on;
plot(xs, mog.pdf(xs, mog.prod(prior_ctr, like_neg)), '-', 'color', color_neg);
plot(xs, mog.pdf(xs, mog.prod(prior_ctr, like_pos)), '-', 'color', color_pos);
subplot(3,3,6); hold on;
plot(xs, mog.pdf(xs, mog.prod(prior_neg, like_ctr)), '-', 'color', color_neg);
plot(xs, mog.pdf(xs, mog.prod(prior_pos, like_ctr)), '-', 'color', color_pos);
subplot(3,3,7);
plot(xs, mog.pdf(xs, like_ctr), '-', 'color', color_ctr);
subplot(3,3,8); hold on;
plot(xs, mog.pdf(xs, like_neg), '-', 'color', color_neg);
plot(xs, mog.pdf(xs, like_pos), '-', 'color', color_pos);
subplot(3,3,9);
plot(xs, mog.pdf(xs, like_ctr), '-', 'color', color_ctr);
