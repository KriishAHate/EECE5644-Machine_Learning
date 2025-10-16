%% Question 3 - Wine Quality Dataset
% Minimum probability of error classifier with Gaussian assumption
clear; close all; clc;

% ---------- Load Dataset ----------
wine_folder = '/Users/kriishhate/Desktop/Fall 2025 Assignments /Intro to machine learning and pattern recognition /Code/wine+quality/';
file = fullfile(wine_folder, 'winequality-white.csv');
opts = detectImportOptions(file, 'Delimiter',';');
T = readtable(file, opts);

X = table2array(T(:, 1:end-1));  % 11 features
y = table2array(T(:, end));       % quality labels (0-10)
[n, d] = size(X);
classes = unique(y);
K = numel(classes);

fprintf('Wine Dataset: %d samples, %d features, %d classes\n', n, d, K);

% ---------- Estimate Parameters from Training Data ----------
% Using all samples from each class with sample averages
mu = zeros(K, d);
Sigma = cell(K, 1);
prior = zeros(K, 1);

for k = 1:K
    % Find samples belonging to class k
    idx = (y == classes(k));
    Xk = X(idx, :);
    
    % Sample estimates
    mu(k,:) = mean(Xk, 1);        % Sample mean
    Sigma{k} = cov(Xk);           % Sample covariance
    prior(k) = sum(idx) / n;      % Sample count for priors
end

% ---------- Regularization (Using Hint) ----------
alpha = 0.01;  % Small regularization parameter
for k = 1:K
    C = Sigma{k};  % Original sample covariance
    
    % Method from hint: λ = α * trace(C) / rank(C)
    % Using arithmetic mean of eigenvalues approach
    lambda = alpha * trace(C) / d;
    
    % Regularize: C_reg = C + λI
    Sigma{k} = C + lambda * eye(d);
    
    fprintf('Class %d: λ = %.6f\n', classes(k), lambda);
end

% ---------- MAP Classification ----------
% Apply minimum P(error) rule on all training samples
pred = zeros(n, 1);

for i = 1:n
    % Compute posterior for each class
    posterior = zeros(K, 1);
    
    for k = 1:K
        % Gaussian likelihood * prior
        likelihood = mvnpdf(X(i,:), mu(k,:), Sigma{k});
        posterior(k) = likelihood * prior(k);
    end
    
    % MAP decision
    [~, idx] = max(posterior);
    pred(i) = classes(idx);
end

% ---------- Error and Confusion Matrix ----------
% Count errors and compute error probability
errors = sum(pred ~= y);
error_prob = errors / n;

% Confusion matrix
confMat = confusionmat(y, pred, 'Order', classes);

fprintf('\n=== RESULTS ===\n');
fprintf('Error probability: %.4f\n', error_prob);
fprintf('Accuracy: %.2f%%\n', 100*(1-error_prob));

% Display confusion matrix
fprintf('\nConfusion Matrix:\n');
disp(confMat);

% ---------- Visualization (2D PCA) ----------
% Using first two principal components
[coeff, score, latent] = pca(X);
var_explained = 100 * latent / sum(latent);

figure;
gscatter(score(:,1), score(:,2), y);
grid on;
title('Wine Quality: 2D PCA Visualization');
xlabel(sprintf('PC1 (%.1f%% variance)', var_explained(1)));
ylabel(sprintf('PC2 (%.1f%% variance)', var_explained(2)));
legend('Location','best');

