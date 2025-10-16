%% Question 1 Part B – Naive Bayes classifier (incorrect covariance)
clear; close all; clc;

% ---- Step 1: Generate data (same as Part A) ----
N = 10000;
p0 = 0.65; p1 = 0.35;
u = rand(1,N) >= p0;
N0 = sum(u==0); N1 = sum(u==1);

% True parameters
mu0 = [-0.5; -0.5; -0.5];
C0 = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
mu1 = [1; 1; 1];
C1 = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];

% Generate data from TRUE distributions
r0 = mvnrnd(mu0, C0, N0);
r1 = mvnrnd(mu1, C1, N1);
X = [r0; r1];
labels = [zeros(N0,1); ones(N1,1)];

% ---- Step 2: Naive Bayes assumes INCORRECT model ----
% Correct means but WRONG covariance (identity matrix)
C_naive = eye(3);  % Assuming independence and unit variance

% ---- Step 3: Compute log-likelihood ratios using NAIVE model ----
% For Naive Bayes with identity covariance:
% log p(x|L=i) = -0.5 * ||x - mu_i||^2 - (d/2)log(2π)
% The constant term cancels in the ratio

logp0_naive = log_mvnpdf(X, mu0', C_naive);
logp1_naive = log_mvnpdf(X, mu1', C_naive);
logLambda_naive = logp1_naive - logp0_naive;

% ---- Step 4: Generate ROC using sorting method ----
[sorted_scores, sort_idx] = sort(logLambda_naive, 'descend');
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

% ---- Step 5: Find optimal threshold ----
Perror = FPR * p0 + (1 - TPR) * p1;
[minErr, idxMin] = min(Perror);
bestTPR = TPR(idxMin);
bestFPR = FPR(idxMin);

if idxMin == 1
    bestLogThreshold = inf;
elseif idxMin == N+1
    bestLogThreshold = -inf;
else
    bestLogThreshold = (sorted_scores(idxMin-1) + sorted_scores(idxMin)) / 2;
end

% ---- Step 6: Plot ROC ----
figure;
plot(FPR, TPR, 'r-', 'LineWidth', 1.5); hold on;
plot(bestFPR, bestTPR, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot([0 1], [0 1], 'k--', 'LineWidth', 0.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Part B: Naive Bayes ROC (Incorrect Covariance Assumption)');
grid on;
legend('NB ROC curve', sprintf('Min P_e = %.4f', minErr), 'Random', 'Location','SouthEast');
axis([0 1 0 1]);

% ---- Step 7: Results ----
fprintf('\n===== PART B: NAIVE BAYES RESULTS =====\n');
fprintf('Model Mismatch: Using I instead of true covariances\n');
fprintf('Minimum probability of error = %.4f\n', minErr);
fprintf('True Positive Rate = %.3f\n', bestTPR);
fprintf('False Positive Rate = %.3f\n', bestFPR);
fprintf('Empirical log(threshold) = %.4f\n', bestLogThreshold);
fprintf('Theoretical log(threshold) = %.4f\n', log(p0/p1));

%% Helper function
function logp = log_mvnpdf(X, mu, Sigma)
    [N, d] = size(X);
    [U, p] = chol(Sigma);
    if p > 0
        error('Covariance not positive definite');
    end
    Q = (X - mu) / U;
    q = sum(Q.^2, 2);
    c = d*log(2*pi) + 2*sum(log(diag(U)));
    logp = -0.5*(c + q);
end
