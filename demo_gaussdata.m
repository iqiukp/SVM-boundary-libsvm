clc
close all
addpath(genpath(pwd))

% generate samples (two-dimensional gaussian distribution)
mu = [5 5];
sigma = [1 0; 0 1];
X_1 = mvnrnd(mu, sigma, 100);
label_1 = ones(100, 1);

mu = [2 10];
sigma = [1 0; 0 1];
X_2 = mvnrnd(mu, sigma, 100);
label_2 = 2*ones(100, 1);

mu = [-2 7];
sigma = [1 0; 0 1];
X_3 = mvnrnd(mu, sigma, 100);
label_3 = 3*ones(100, 1);

% 
data = [X_1; X_2; X_3];
label = [label_1; label_2; label_3];

% parameter setting
c = 0.8;  
g = 0.05; 

% train svm model
cmd = ['-s 0 -t 2 ', '-c ', num2str(c), ' -g ', num2str(g), ' -q'];
model = libsvmtrain(label, data, cmd);
[~, acc, ~] = libsvmpredict(label, data, model); 

% meshgrid
d = 0.02;
[X1, X2] = meshgrid(min(data(:,1)):d:max(data(:,1)), min(data(:,2)):d:max(data(:,2)));
X_grid = [X1(:), X2(:)];
% set grid point label (only as input parameter)
grid_label = ones(size(X_grid, 1), 1);
% predict grid point labels
[pre_label, ~, ~] = libsvmpredict(grid_label, X_grid, model);

% 
figure
% set(gcf,'position',[300 150 420 360])
color_p = [150, 138, 191;12, 112, 104; 220, 94, 75]/255; 
color_b = [218, 216, 232; 179, 226, 219; 244, 195, 171]/255; 
hold on
ax(1:3) = gscatter(X_grid (:,1), X_grid (:,2), pre_label, color_b);
legend('off')
axis tight

% 
ax(4:6) = gscatter(data(:,1), data(:,2), label);
set(ax(4), 'Marker','o', 'MarkerSize', 6, 'MarkerEdgeColor','k', 'MarkerFaceColor', color_p(1,:));
set(ax(5), 'Marker','o', 'MarkerSize', 6, 'MarkerEdgeColor','k', 'MarkerFaceColor', color_p(2,:));
set(ax(6), 'Marker','o', 'MarkerSize', 6, 'MarkerEdgeColor','k', 'MarkerFaceColor', color_p(3,:));
legend('off')
set(gca, 'linewidth', 1.1)
title('Decision boundary')
axis tight