% Question 3 HAR Dataset 
clear; close all; clc;

% ---------- Load ----------
folder = 'UCI HAR Dataset';  
X = load(fullfile(folder, 'train', 'X_train.txt'));
y = load(fullfile(folder, 'train', 'y_train.txt'));

[n, d]   = size(X);
classes  = unique(y);
K        = numel(classes);
fprintf('HAR: %d samples, %d features, %d classes\n', n, d, K);

% ---------- Per-class estimates ----------
mu    = zeros(K, d);
Sigma = cell(K,1);
prior = zeros(K,1);

for k = 1:K
    idx      = (y == classes(k));
    Xk       = X(idx,:);
    mu(k,:)  = mean(Xk,1);
    C        = cov(Xk) + 1e-12*eye(d);   % tiny jitter

    % ----- Regularization (assignment hint) -----
    % Choose ONE of the two λ formulas below.
    alpha  = 0.05;                         % 0 < alpha << 1 

    % (A) Arithmetic mean of eigenvalues: λ = α * trace(C)/d
    lambda = alpha * trace(C) / d;

    % (B) Optional variant in the hint: λ = α * trace(C)/rank(C)
    % lambda = alpha * trace(C) / max(1, rank(C));

    Creg      = C + lambda*eye(d);         % C_reg = C + λ I
    Sigma{k}  = Creg;
    prior(k)  = nnz(idx)/n;

    fprintf('Class %d: lambda = %.3e (alpha=%.3g)\n', classes(k), lambda, alpha);
end

% ---------- Precompute Cholesky + log|Σ| ----------
cholS   = cell(K,1);
logdetS = zeros(K,1);
for k = 1:K
    [R,p] = chol(Sigma{k});
    if p > 0
        bump = 1e-6 * trace(Sigma{k})/d;
        [R,~] = chol(Sigma{k} + bump*eye(d));
        Sigma{k} = Sigma{k} + bump*eye(d);
    end
    cholS{k}   = R;
    logdetS(k) = 2*sum(log(diag(R)));
end

% ---------- MAP classification (log-domain) ----------
const    = -0.5*d*log(2*pi);
logprior = log(prior + eps);
pred     = zeros(n,1);

for i = 1:n
    xi = X(i,:);
    logpost = zeros(K,1);
    for k = 1:K
        diff = xi - mu(k,:);
        % Mahalanobis via triangular solves with Cholesky: Σ = R'R
        z = diff / cholS{k}';   % solve R' z = diff
        z = z  / cholS{k};      % solve R  w = z
        maha = z*z.';           % ||w||^2
        loglike    = const - 0.5*logdetS(k) - 0.5*maha;
        logpost(k) = logprior(k) + loglike;
    end
    [~, ix] = max(logpost);
    pred(i) = classes(ix);
end

% ---------- Results ----------
errors  = sum(pred ~= y);
errProb = errors / n;
confMat = confusionmat(y, pred, 'Order', classes);

fprintf('\n=== HAR (training set) ===\n');
fprintf('Error probability: %.4f\n', errProb);
fprintf('Accuracy: %.2f%%\n', 100*(1 - errProb));
disp('Confusion matrix (rows=true, cols=pred):');
disp(confMat);

% ---------- 2D PCA ----------
[coeff, score, latent] = pca(X);
varExpl = 100*latent/sum(latent);
figure; gscatter(score(:,1), score(:,2), y);
grid on; title('HAR — 2D PCA (training)');
xlabel(sprintf('PC1 (%.1f%%)', varExpl(1)));
ylabel(sprintf('PC2 (%.1f%%)', varExpl(2)));
legend('Location','bestoutside');
