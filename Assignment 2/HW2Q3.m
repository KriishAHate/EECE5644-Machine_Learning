%% Q3 MAP Estimation for Vehicle Localization 
clear; close all; clc;

% Problem parameters as specified
sigma_i = 0.3;  % Measurement noise standard deviation (given)
sigma_x = 0.25; % Prior standard deviation for x (suggested)
sigma_y = 0.25; % Prior standard deviation for y (suggested)

% Set true vehicle position (must be inside unit circle)
% change this to test different scenarios
x_true = 0.3;
y_true = 0.4;
fprintf('True vehicle position: (%.2f, %.2f)\n\n', x_true, y_true);

% Create grid for evaluation (-2 to 2 as specified)
resolution = 0.02;
[X, Y] = meshgrid(-2:resolution:2, -2:resolution:2);

% Store results for analysis
MAP_estimates = zeros(4, 2);
MAP_errors = zeros(4, 1);

% First pass: Find global min/max for consistent contour levels
J_all = cell(4, 1);
measurements_all = cell(4, 1);
landmarks_all = cell(4, 1);

for K = 1:4
    % Place K landmarks evenly on unit circle
    angles = linspace(0, 2*pi, K+1);
    angles = angles(1:K);  % Remove duplicate
    landmarks = [cos(angles); sin(angles)]';
    landmarks_all{K} = landmarks;
    
    % Generate true distances
    true_distances = sqrt((x_true - landmarks(:,1)).^2 + ...
                          (y_true - landmarks(:,2)).^2);
    
    % Generate noisy measurements (ensuring non-negative)
    measurements = true_distances + sigma_i * randn(K, 1);
    while any(measurements < 0)
        neg_idx = measurements < 0;
        measurements(neg_idx) = true_distances(neg_idx) + ...
                               sigma_i * abs(randn(sum(neg_idx), 1));
    end
    measurements_all{K} = measurements;
    
    % Evaluate MAP objective function
    J = zeros(size(X));
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            x = X(i, j);
            y = Y(i, j);
            
            % Distances from candidate position to landmarks
            d = sqrt((x - landmarks(:,1)).^2 + (y - landmarks(:,2)).^2);
            
            % MAP objective (simplified, constants removed)
            J(i, j) = sum((measurements - d).^2 / (2*sigma_i^2)) + ...
                      x^2/(2*sigma_x^2) + y^2/(2*sigma_y^2);
        end
    end
    J_all{K} = J;
    
    % Find MAP estimate
    [min_val, min_idx] = min(J(:));
    [row, col] = ind2sub(size(J), min_idx);
    MAP_estimates(K, :) = [X(row, col), Y(row, col)];
    MAP_errors(K) = sqrt((MAP_estimates(K, 1) - x_true)^2 + ...
                         (MAP_estimates(K, 2) - y_true)^2);
end

% Find consistent contour levels
J_min = inf; J_max = -inf;
for K = 1:4
    J_min = min(J_min, min(J_all{K}(:)));
    J_max = max(J_max, max(J_all{K}(:)));
end
contour_levels = linspace(J_min, J_min + 15, 20);

% Create figure with consistent contour plots
figure('Position', [100, 100, 1200, 900]);

for K = 1:4
    subplot(2, 2, K);
    
    % Plot consistent contours
    contour(X, Y, J_all{K}, contour_levels, 'LineWidth', 1);
    colorbar;
    hold on;
    
    % Mark true position (+ as specified)
    plot(x_true, y_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
    
    % Mark MAP estimate 
    plot(MAP_estimates(K, 1), MAP_estimates(K, 2), 'g*', ...
         'MarkerSize', 12, 'LineWidth', 2);
    
    % Mark landmarks (o as specified)
    plot(landmarks_all{K}(:,1), landmarks_all{K}(:,2), 'ko', ...
         'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'k');
    
    % Show unit circle
    theta = linspace(0, 2*pi, 100);
    plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1);
    
    % Labels and formatting
    xlabel('x');
    ylabel('y');
    title(sprintf('K = %d Landmarks\nMAP Error = %.3f', K, MAP_errors(K)));
    axis equal;
    xlim([-2 2]);
    ylim([-2 2]);
    grid on;
    
    % Add legend for first subplot
    if K == 1
        legend('Objective Contours', 'True Position (+)', ...
               'MAP Estimate (*)', 'Landmarks (o)', 'Unit Circle', ...
               'Location', 'northeast');
    end
end

subtitle('MAP Position Estimation with Consistent Contour Levels');

%% Print analysis results
fprintf('=== MAP ESTIMATION RESULTS ===\n\n');
fprintf('Prior parameters: σ_x = %.2f, σ_y = %.2f\n', sigma_x, sigma_y);
fprintf('Measurement noise: σ_i = %.2f\n\n', sigma_i);

fprintf('Number of | MAP Estimate  | True Position | Error\n');
fprintf('Landmarks |   (x, y)      |    (x, y)     | \n');
fprintf('----------|---------------|---------------|-------\n');
for K = 1:4
    fprintf('    %d     | (%.3f, %.3f) | (%.3f, %.3f) | %.4f\n', ...
            K, MAP_estimates(K, 1), MAP_estimates(K, 2), ...
            x_true, y_true, MAP_errors(K));
end

%% Analysis comments
fprintf('\n=== ANALYSIS ===\n');
fprintf('1. As K increases from 1 to 4, the MAP error generally decreases.\n');
fprintf('2. The contours become more circular and concentrated around the true position.\n');
fprintf('3. With K=1, the estimate has high uncertainty (elongated contours).\n');
fprintf('4. With K=4, the objective function is more peaked (tighter contours).\n');
fprintf('5. The prior pulls estimates toward the origin when measurements are weak.\n');

%% Additional visualization - Error vs K
figure;
plot(1:4, MAP_errors, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Landmarks (K)');
ylabel('MAP Estimation Error');
title('MAP Estimation Error vs Number of Landmarks');
grid on;
xlim([0.5 4.5]);