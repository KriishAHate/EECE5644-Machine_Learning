function revisedGMMModelSelection
%% EECE5644 Assignment 3 - Question 2 GMMModelSelection
% Robust 10-fold CV GMM model-order selection experiment.
% Evaluates K = 1 to 10 on datasets of size N = [10, 100, 1000]
% Repeats each experiment 100 times and reports top 6 selected orders.

rng(0); % For reproducibility

%% Step 1: Define true 4-component 2D GMM with overlap
alpha = [0.2, 0.3, 0.3, 0.2];
mu = [0 0; 1 1; 5 5; -3 -3];
Sigma(:,:,1) = [1 0.5; 0.5 1];
Sigma(:,:,2) = [1 0.5; 0.5 1];
Sigma(:,:,3) = [0.5 0; 0 0.5];
Sigma(:,:,4) = [1 0; 0 1];
gmtrue = gmdistribution(mu, Sigma, alpha);

%% Step 2: Experiment parameters
datasetSizes = [10, 100, 1000];
maxK = 10;
trials = 100;
options = statset('MaxIter', 2000, 'TolFun', 1e-6, 'Display', 'off');
replicates = 5;
regularization = 1e-6;

countsAll = zeros(length(datasetSizes), maxK);
failedRepeatsAll = zeros(length(datasetSizes), 1);

%% Step 3: Run experiments
for ds = 1:length(datasetSizes)
    N = datasetSizes(ds);
    fprintf('\nRunning N = %d, trials = %d...\n', N, trials);
    counts = zeros(1, maxK);
    failed = 0;

    for t = 1:trials
        X = random(gmtrue, N);
        scores = zeros(1, maxK);

        for k = 1:maxK
            scores(k) = crossval_gmm(X, k, options, replicates, regularization);
        end

        if all(~isfinite(scores))
            failed = failed + 1;
            continue;
        end

        [~, bestK] = max(scores);
        counts(bestK) = counts(bestK) + 1;
    end

    countsAll(ds, :) = counts;
    failedRepeatsAll(ds) = failed;
    fprintf('N = %d done. Failed trials: %d/%d\n', N, failed, trials);
end

%% Step 4: Plot selection frequencies
figure('Units','normalized','Position',[0.1 0.1 0.6 0.75]);
for i = 1:length(datasetSizes)
    subplot(length(datasetSizes),1,i);
    bar(1:maxK, countsAll(i,:), 'FaceColor', [0.2 0.5 0.8]);
    xlabel('Number of Components K');
    ylabel('Selection Count');
    title(sprintf('Selection Frequency (N = %d, Trials = %d, Fails = %d)', ...
        datasetSizes(i), trials, failedRepeatsAll(i)));
    xticks(1:maxK);
    ylim([0, max(countsAll(i,:)) + 5]);
    grid on;
end

%% Step 5: Report top 6 selected orders
fprintf('\nTop 6 Selected GMM Orders per Dataset Size:\n');
for i = 1:length(datasetSizes)
    [sortedCounts, sortedK] = sort(countsAll(i,:), 'descend');
    topK = sortedK(1:6);
    topCounts = sortedCounts(1:6);
    fprintf('N = %d:\n', datasetSizes(i));
    fprintf('  Top 6 Ks     = %s\n', mat2str(topK));
    fprintf('  Selection #s = %s\n\n', mat2str(topCounts));
end

% Optional: Save results
save('revisedGMMModelSelection_results.mat', 'countsAll', 'failedRepeatsAll', 'datasetSizes', 'trials');

end

%% Helper Function: Cross-validated log-likelihood for GMM
function avgLL = crossval_gmm(X, K, options, replicates, regularization)
N = size(X,1);
folds = min(10, N);
cv = cvpartition(N, 'KFold', folds);
logL = -Inf(folds,1);

for f = 1:folds
    Xtrain = X(training(cv,f), :);
    Xtest  = X(test(cv,f), :);

    if size(Xtrain,1) <= K
        continue;
    end

    try
        gm = fitgmdist(Xtrain, K, ...
            'Options', options, ...
            'RegularizationValue', regularization, ...
            'Replicates', replicates, ...
            'Start', 'plus', ...
            'CovarianceType', 'full', ...
            'SharedCovariance', false);

        ll = sum(log(pdf(gm, Xtest)));
        if isfinite(ll)
            logL(f) = ll;
        end
    catch
        % EM failed — leave logL(f) as -Inf
    end
end

finiteIdx = isfinite(logL);
if any(finiteIdx)
    avgLL = mean(logL(finiteIdx));
else
    avgLL = -Inf;
end
end
