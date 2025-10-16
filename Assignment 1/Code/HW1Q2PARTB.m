%% QUESTION 2 - PART B: ASYMMETRIC LOSS MATRIX CLASSIFICATION
clear all; close all; clc;

% ---------- Step 1: Problem setup (same as Part A) ----------
C = 4; % number of classes
N = 10000; % total samples
n = 2; % data dimension (2D for plotting)

% Class priors (uniform)
gmmParameters.priors = ones(1,C)/C;

% Means (spread on x-axis)
gmmParameters.meanVectors = [-30, -10, 10, 30;
                               0,   0,  0,  0];

% Covariances (PSD by construction)
for l = 1:C
    A = 5*[4 1; 1 4] + 0.5*rand(n,n);
    gmmParameters.covMatrices(:,:,l) = A'*A;
end

% ---------- Step 2: Generate data from the GMM ----------
[x, labels] = generateDataFromGMM(N, gmmParameters);

% Class counts
for l = 1:C
    Nclass(l,1) = length(find(labels == l));
end

% ---------- Step 3: ERM classifier with ASYMMETRIC loss ----------
% Likelihoods p(x|L=l) for all samples
pxgivenl = zeros(C, N);
for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF(x, gmmParameters.meanVectors(:,l), ...
                                    gmmParameters.covMatrices(:,:,l));
end

% Marginal p(x) = sum_l P(L=l) p(x|L=l)
px = gmmParameters.priors * pxgivenl;

% Posteriors P(L=l|x) via Bayes' rule
classPosteriors = pxgivenl .* repmat(gmmParameters.priors',1,N) ./ repmat(px, C, 1);

% PART B: Asymmetric loss matrix (high penalty for misclassifying class 4)
lossMatrix = [0  10  10  100;
              1   0  10  100;
              1   1   0  100;
              1   1   1    0];

% Expected risk R(d|x) = sum_l Loss(d,l) P(L=l|x)
expectedRisks = lossMatrix * classPosteriors;

% Decision: argmin_d R(d|x)
[~, decisions] = min(expectedRisks, [], 1);

% ---------- Step 4: Confusion matrix ----------
ConfusionMatrix = zeros(C);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions==d & labels==l);
        ConfusionMatrix(d,l) = length(ind_dl) / length(find(labels==l));
    end
end

% ---------- Step 5: Calculate Minimum Expected Risk ----------
% Empirical risk = average loss over all samples
empiricalRisk = 0;
for i = 1:N
    true_label = labels(i);
    decision = decisions(i);
    empiricalRisk = empiricalRisk + lossMatrix(decision, true_label);
end
empiricalRisk = empiricalRisk / N;

% ---------- Step 6: Display Results ----------
fprintf('====== PART B RESULTS: ASYMMETRIC LOSS MATRIX ======\n\n');
fprintf('Loss Matrix (rows=decisions, cols=true labels):\n');
fprintf('     L=1   L=2   L=3   L=4\n');
for d = 1:C
    fprintf('D=%d  ', d);
    for l = 1:C
        fprintf('%3d   ', lossMatrix(d,l));
    end
    fprintf('\n');
end

fprintf('\nConfusion Matrix P(D=i|L=j):\n');
fprintf('     L=1    L=2    L=3    L=4\n');
for d = 1:C
    fprintf('D=%d  ', d);
    for l = 1:C
        fprintf('%.3f  ', ConfusionMatrix(d,l));
    end
    fprintf('\n');
end

fprintf('\n*** MINIMUM EXPECTED RISK (empirical): %.4f ***\n', empiricalRisk);

% Calculate how often class 4 is correctly classified
class4_accuracy = ConfusionMatrix(4,4);
fprintf('\nClass 4 accuracy: %.1f%% (should be high due to loss=100 for misclassification)\n', ...
        100*class4_accuracy);

% ---------- Step 7: Visualization ----------
mShapes = '.o^s'; 
figure('Name', 'Part B: Asymmetric Loss Classification');
hold on;

for l = 1:C
    correct_ind = find(labels==l & decisions==l);
    incorrect_ind = find(labels==l & decisions~=l);
    
    % Correct (green)
    if ~isempty(correct_ind)
        plot(x(1,correct_ind), x(2,correct_ind), mShapes(l), ...
             'MarkerEdgeColor','g','MarkerFaceColor','g', ...
             'MarkerSize',6,'DisplayName',sprintf('Class %d - Correct', l));
    end
    
    % Incorrect (red)
    if ~isempty(incorrect_ind)
        plot(x(1,incorrect_ind), x(2,incorrect_ind), mShapes(l), ...
             'MarkerEdgeColor','r','MarkerFaceColor','r', ...
             'MarkerSize',6,'DisplayName',sprintf('Class %d - Incorrect', l));
    end
end

hold off;
axis equal; grid on;
xlabel('Feature 1 (X_1)');
ylabel('Feature 2 (X_2)');
title(sprintf('Part B: Asymmetric Loss (Min Risk = %.2f)', empiricalRisk));
legend('Location','best');

% ---------- Additional Analysis ----------
fprintf('\n====== Per-class Accuracy ======\n');
for l = 1:C
    fprintf('Class %d accuracy: %.1f%%\n', l, 100*ConfusionMatrix(l,l));
end

fprintf('\n====== Impact of Asymmetric Loss ======\n');
fprintf('We can see how the classifier strongly favors deciding class 4\n');
fprintf('to avoid the high penalty (loss=100) for misclassifying true class 4 samples.\n');