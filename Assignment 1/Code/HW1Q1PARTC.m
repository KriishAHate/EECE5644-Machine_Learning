%% Question 1 Part C – Fisher LDA Classifier
clear; close all; clc;

% ---- Step 1: Generate the same data as Parts A & B ----
N = 10000;
p0 = 0.65; p1 = 0.35;
u = rand(1,N) >= p0;
N0 = sum(u==0); N1 = sum(u==1);

% True parameters (for data generation)
mu0_true = [-0.5; -0.5; -0.5];
C0_true = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
mu1_true = [1; 1; 1];
C1_true = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];

% Generate data
r0 = mvnrnd(mu0_true, C0_true, N0);
r1 = mvnrnd(mu1_true, C1_true, N1);
X = [r0; r1];
labels = [zeros(N0,1); ones(N1,1)];

% ---- Step 2: ESTIMATE parameters from samples ----
% Sample means
mu0_est = mean(r0)';  % Estimated mean for class 0
mu1_est = mean(r1)';  % Estimated mean for class 1

% Sample covariances
C0_est = cov(r0);     % Estimated covariance for class 0
C1_est = cov(r1);     % Estimated covariance for class 1

% ---- Step 3: Compute Fisher LDA projection vector ----
% Within-class scatter matrix (using equal weights as instructed)
Sw = (C0_est + C1_est) / 2;

% Between-class scatter matrix
mu_diff = mu1_est - mu0_est;
Sb = mu_diff * mu_diff';

% Fisher LDA projection vector
[V, D] = eig(Sb, Sw);  % Generalized eigendecomposition
[~, idx] = max(diag(D));
w_LDA = V(:, idx);

% Ensure consistent orientation
if w_LDA' * mu_diff < 0
    w_LDA = -w_LDA;
end

% ---- Step 4: Project data onto LDA direction ----
X_projected = X * w_LDA;  % N×1 vector of projections

% ---- Step 5: Generate ROC by sorting projected values ----
[sorted_scores, sort_idx] = sort(X_projected, 'descend');
sorted_labels = labels(sort_idx);

TPR = zeros(N+1, 1);
FPR = zeros(N+1, 1);
TPR(1) = 0; FPR(1) = 0;

P = sum(labels == 1);
N_neg = sum(labels == 0);
TP = 0; FP = 0;

for i = 1:N
    if sorted_labels(i) == 1
        TP = TP + 1;
    else
        FP = FP + 1;
    end
    TPR(i+1) = TP / P;
    FPR(i+1) = FP / N_neg;
end

% ---- Step 6: Find optimal threshold ----
Perror = FPR * p0 + (1 - TPR) * p1;
[minErr, idxMin] = min(Perror);
bestTPR = TPR(idxMin);
bestFPR = FPR(idxMin);

if idxMin == 1
    bestThreshold = inf;
elseif idxMin == N+1
    bestThreshold = -inf;
else
    bestThreshold = (sorted_scores(idxMin-1) + sorted_scores(idxMin)) / 2;
end

% ---- Step 7: Plot ROC ----
figure;
plot(FPR, TPR, 'g-', 'LineWidth', 1.5); hold on;
plot(bestFPR, bestTPR, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot([0 1], [0 1], 'k--', 'LineWidth', 0.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Part C: Fisher LDA ROC Curve');
grid on;
legend('LDA ROC', sprintf('Min P_e = %.4f', minErr), 'Random', 'Location', 'SouthEast');
axis([0 1 0 1]);

% ---- Step 8: Display results ----
fprintf('\n===== PART C: FISHER LDA RESULTS =====\n');
fprintf('Using ESTIMATED parameters from samples\n');
fprintf('---------------------------------------\n');
fprintf('Fisher LDA projection vector:\n');
fprintf('  w_LDA = [%.4f, %.4f, %.4f]^T\n', w_LDA);
fprintf('\nClassifier Performance:\n');
fprintf('  Minimum probability of error = %.4f\n', minErr);
fprintf('  True Positive Rate = %.3f\n', bestTPR);
fprintf('  False Positive Rate = %.3f\n', bestFPR);
fprintf('  Optimal threshold on projected axis = %.4f\n', bestThreshold);