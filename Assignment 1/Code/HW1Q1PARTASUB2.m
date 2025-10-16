%% Question 1 Part A.2 – Optimal Bayesian classifier (true parameters)
clear; close all; clc;

% ---------- Step 1: Given problem parameters ----------
N = 10000; % number of samples
p0 = 0.65; p1 = 0.35; % class priors

% True means and covariance matrices
mu0 = [-0.5; -0.5; -0.5];
C0 = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
mu1 = [1; 1; 1];
C1 = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];

% ---------- Step 2: Generate class labels according to priors ----------
u = rand(1,N) >= p0; % 0 if random < p0, 1 if >= p0
N0 = sum(u==0); N1 = sum(u==1);

% Generate samples for each class
r0 = mvnrnd(mu0, C0, N0);
r1 = mvnrnd(mu1, C1, N1);

% Combine into full dataset
X = [r0; r1];
labels = [zeros(N0,1); ones(N1,1)];

% ---------- Step 3: Compute true log-likelihood ratio ----------
logp0 = log_mvnpdf(X, mu0', C0);
logp1 = log_mvnpdf(X, mu1', C1);
logLambda = logp1 - logp0; % log-likelihood ratio

% ---------- Step 4: Generate ROC by sorting scores ----------
% This is the standard way to generate ROC curves
[sorted_scores, sort_idx] = sort(logLambda, 'descend');
sorted_labels = labels(sort_idx);

% Calculate TPR and FPR for each possible threshold
TPR = zeros(N+1, 1);
FPR = zeros(N+1, 1);

% Start with everything classified as negative (threshold = +inf)
TPR(1) = 0;
FPR(1) = 0;

% Count total positives and negatives
P = sum(labels == 1);  % Total actual positives
N_neg = sum(labels == 0);  % Total actual negatives

% Accumulate TP and FP as threshold decreases
TP = 0;
FP = 0;

for i = 1:N
    if sorted_labels(i) == 1
        TP = TP + 1;
    else
        FP = FP + 1;
    end
    TPR(i+1) = TP / P;
    FPR(i+1) = FP / N_neg;
end

% ---------- Step 5: Find optimal threshold for min P(error) ----------
Perror = FPR * p0 + (1 - TPR) * p1;
[minErr, idxMin] = min(Perror);
bestTPR = TPR(idxMin);
bestFPR = FPR(idxMin);

% Find the actual threshold value at this point
if idxMin == 1
    bestLogThreshold = inf;
elseif idxMin == N+1
    bestLogThreshold = -inf;
else
    bestLogThreshold = (sorted_scores(idxMin-1) + sorted_scores(idxMin)) / 2;
end

% Theoretical threshold
threshold_theory = log(p0/p1);

% ---------- Step 6: Plot ROC ----------
figure;
plot(FPR, TPR, 'b', 'LineWidth', 1.5); hold on;
plot(bestFPR, bestTPR, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot([0 1], [0 1], 'k--', 'LineWidth', 0.5); % diagonal reference line
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('Question 1 Part A – ROC Curve (True Gaussian Model)');
grid on;
legend('ROC curve', sprintf('Min P_e = %.4f', minErr), 'Random Guess', 'Location','SouthEast');
axis([0 1 0 1]);

% ---------- Step 7: Display results ----------
fprintf('--- Part A Results ---\n');
fprintf('Theoretical log(threshold) = %.4f\n', threshold_theory);
fprintf('Empirical log(threshold) = %.4f\n', bestLogThreshold);
fprintf('Minimum probability of error = %.4f\n', minErr);
fprintf('At optimal threshold:\n');
fprintf('  True Positive Rate = %.3f\n', bestTPR);
fprintf('  False Positive Rate = %.3f\n', bestFPR);

%% ---------- Function for log multivariate normal PDF ----------
function logp = log_mvnpdf(X, mu, Sigma)
    % X: N×d, mu: 1×d, Sigma: d×d
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