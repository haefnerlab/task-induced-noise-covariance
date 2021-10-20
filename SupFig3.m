save_file = 'Haefner2016.mat';
data = load(save_file, 'Projection');
projective_fields = data.Projection.G;

[n_pixels, n_neurons] = size(projective_fields);
random_images = randn(n_pixels, 1000);
ff_drive = random_images' * projective_fields;

[~,s,v] = svd(cov(ff_drive));
s = diag(s);

n_highlight = 5;
colors = lines(n_highlight);

% Swap colors 1 and 2 to match coloring of original eigenvectors above
tmp = colors(1,:);
colors(1,:) = colors(2,:);
colors(2,:) = tmp;

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