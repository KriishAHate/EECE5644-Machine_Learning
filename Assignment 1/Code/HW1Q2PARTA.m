% QUESTION 2 - PART A: MINIMUM PROBABILITY OF ERROR CLASSIFICATION
% MAP rule with 0–1 loss for a 4-class GMM
clear all; close all;

% ---------- Step 1: Problem setup ----------
C = 4;          % number of classes
N = 10000;      % total samples
n = 2;          % data dimension (2D for plotting)

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
% Draw N samples with labels according to the priors
[x, labels] = generateDataFromGMM(N, gmmParameters);

% Class counts (≈ N/4 each due to uniform priors)
for l = 1:C
    Nclass(l,1) = length(find(labels == l));
end

% ---------- Step 3: Minimum-error (MAP) classifier under 0–1 loss ----------
% (a) Likelihoods p(x|L=l) for all samples
pxgivenl = zeros(C, N);
for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF( ...
        x, gmmParameters.meanVectors(:,l), gmmParameters.covMatrices(:,:,l));
end

% (b) Marginal p(x) = sum_l P(L=l) p(x|L=l)
px = gmmParameters.priors * pxgivenl;

% (c) Posteriors P(L=l|x) via Baye's rule
classPosteriors = pxgivenl .* repmat(gmmParameters.priors',1,N) ./ repmat(px, C, 1);

% (d) 0–1 loss
lossMatrix = [0 1 1 1;
              1 0 1 1;
              1 1 0 1;
              1 1 1 0];

% (e) Expected risk R(d|x) = sum_l Loss(d,l) P(L=l|x)
expectedRisks = lossMatrix * classPosteriors;

% (f) Decision: argmin_d R(d|x)  (== MAP for 0–1 loss)
[~, decisions] = min(expectedRisks, [], 1);

% ---------- Step 4: Confusion matrix ----------
% ConfusionMatrix(d,l) = P(D=d | L=l)
ConfusionMatrix = zeros(C);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions==d & labels==l);
        ConfusionMatrix(d,l) = length(ind_dl) / length(find(labels==l));
    end
end

disp('====== PART A RESULTS: MAP CLASSIFICATION (0-1 LOSS) ======');
disp('Confusion Matrix P(D=i|L=j):');
disp('Rows = Decisions, Columns = True Labels');
disp('Perfect classifier would have identity matrix');
ConfusionMatrix

% Overall error (uniform priors - mean diagonal complement)
% Overall probability of error (prior-weighted) 
Pcorrect = gmmParameters.priors * diag(ConfusionMatrix);
Perror   = 1 - Pcorrect;
fprintf('Overall Probability of Error (prior-weighted): %.4f\n', Perror);

% ---------- Step 5: Visualization ----------
% Green = correct, Red = incorrect; markers distinguish true class
mShapes = '.o^s';           % per-class marker symbols
figure(1); clf; hold on;

for l = 1:C
    correct_ind   = find(labels==l & decisions==l);
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
title('Part A: MAP Classification (Green = Correct, Red = Incorrect)');
legend('Location','best');

% ---------- Step 6: Additional stats ----------
fprintf('\n====== Class Distribution ======\n');
fprintf('(Uniform priors, N = %d)\n', N);
for l = 1:C
    fprintf('Class %d: %d samples (%.1f%%)\n', l, Nclass(l), 100*Nclass(l)/N);
end

fprintf('\n====== Per-class Accuracy ======\n');
for l = 1:C
    fprintf('Class %d accuracy: %.1f%%\n', l, 100*ConfusionMatrix(l,l));
end
