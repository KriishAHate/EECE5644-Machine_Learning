%% Question 1 Part A.3 – Determine optimal threshold and minimum P(error)
clear; close all; clc;

% ---------- Load or regenerate the data from Part A ----------
N = 10000; 
p0 = 0.65; p1 = 0.35;

% True parameters
mu0 = [-0.5; -0.5; -0.5];
C0 = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
mu1 = [1; 1; 1];
C1 = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];

% Generate data 
u = rand(1,N) >= p0;
N0 = sum(u==0); N1 = sum(u==1);
r0 = mvnrnd(mu0, C0, N0);
r1 = mvnrnd(mu1, C1, N1);
X = [r0; r1];
labels = [zeros(N0,1); ones(N1,1)];

% Compute log-likelihood ratios
logp0 = log_mvnpdf(X, mu0', C0);
logp1 = log_mvnpdf(X, mu1', C1);
logLambda = logp1 - logp0;

% ---------- Generate ROC data ----------
[sorted_scores, sort_idx] = sort(logLambda, 'descend');
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

% ---------- PART A.3: Find optimal threshold ----------
% Calculate probability of error for each threshold
% P(error; γ) = P(D=1|L=0; γ)P(L=0) + P(D=0|L=1; γ)P(L=1)
Perror = FPR * p0 + (1 - TPR) * p1;

% Find minimum error
[minErr, idxMin] = min(Perror);

% Extract performance at optimal point
optimal_TPR = TPR(idxMin);
optimal_FPR = FPR(idxMin);

% Determine threshold value
if idxMin == 1
    optimal_log_threshold = inf;
    optimal_gamma = inf;
elseif idxMin == N+1
    optimal_log_threshold = -inf;
    optimal_gamma = 0;
else
    optimal_log_threshold = (sorted_scores(idxMin-1) + sorted_scores(idxMin)) / 2;
    optimal_gamma = exp(optimal_log_threshold);
end

% Theoretical values
theoretical_gamma = p0/p1;
theoretical_log_gamma = log(theoretical_gamma);

% ---------- Plot ROC with optimal point highlighted ----------
figure('Position', [100, 100, 600, 600]);
plot(FPR, TPR, 'b-', 'LineWidth', 2); hold on;

% Superimpose optimal point with different marker
plot(optimal_FPR, optimal_TPR, 'rs', 'MarkerSize', 12, ...
     'MarkerFaceColor', 'r', 'LineWidth', 2);

plot([0 1], [0 1], 'k--', 'LineWidth', 1);

xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('Part A.3: ROC Curve with Minimum P(error) Operating Point');
grid on; axis square; axis([0 1 0 1]);

legend('ROC Curve', ...
       sprintf('Min P(error) = %.4f\n(TPR=%.3f, FPR=%.3f)', ...
               minErr, optimal_TPR, optimal_FPR), ...
       'Random Classifier', ...
       'Location', 'SouthEast');

% ---------- Calculate and report results ----------
fprintf('===== PART A.3 RESULTS =====\n\n');

fprintf('Optimal Threshold Determination:\n');
fprintf('--------------------------------\n');
fprintf('Empirical γ that minimizes P(error): %.4f\n', optimal_gamma);
fprintf('Theoretical γ* from priors and 0-1 loss: %.4f\n', theoretical_gamma);
fprintf('Comparison:\n');
fprintf('  Empirical log(γ): %.4f\n', optimal_log_threshold);
fprintf('  Theoretical log(γ*): %.4f\n', theoretical_log_gamma);
fprintf('  Difference in log space: %.4f\n\n', abs(optimal_log_threshold - theoretical_log_gamma));

fprintf('Minimum P(error) Classifier Performance:\n');
fprintf('----------------------------------------\n');
fprintf('Minimum probability of error: %.4f\n', minErr);
fprintf('True Positive Rate: %.4f\n', optimal_TPR);
fprintf('False Positive Rate: %.4f\n\n', optimal_FPR);

fprintf('My Error Probablity Calculation:\n');
fprintf('----------------------------\n');
fprintf('P(error) = P(D=1|L=0)×P(L=0) + P(D=0|L=1)×P(L=1)\n');
fprintf('         = %.4f × %.2f + %.4f × %.2f\n', ...
        optimal_FPR, p0, (1-optimal_TPR), p1);
fprintf('         = %.4f + %.4f\n', ...
        optimal_FPR*p0, (1-optimal_TPR)*p1);
fprintf('         = %.4f\n', minErr);

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