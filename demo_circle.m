clc
close all
addpath(genpath(pwd))

load circle
%
label = data(:,3);
data = data (:, 1:2);

% parameter setting
c = 0.9;  
g = 0.5;

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
color_p = [150, 138, 191; 220, 94, 75]/255; 
color_b = [218, 216, 232; 244, 195, 171]/255; 
hold on
gscatter(X_grid (:,1), X_grid (:,2), pre_label, color_b);
legend('off')
axis tight
% 
ax(3:4) = gscatter(data(:,1), data(:,2), label);
set(ax(3), 'Marker','o', 'MarkerSize', 6, 'MarkerEdgeColor','k', 'MarkerFaceColor', color_p(1,:));
set(ax(4), 'Marker','o', 'MarkerSize', 6, 'MarkerEdgeColor','k', 'MarkerFaceColor', color_p(2,:));

legend('off')
set(gca, 'linewidth', 1.1)
title('Decision boundary')
axis tight