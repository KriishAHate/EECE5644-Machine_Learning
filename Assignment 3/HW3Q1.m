%% EECE5644 Assignment 3 - Question 1 
clear; clc; close all; rng(0);

%% ---------------- Problem setup ----------------
n = 3; C = 4;
mu{1} = [-1.0; 0.0; 0.0];
mu{2} = [0.8; 0.5; 0.2];
mu{3} = [-0.2; -0.9; 0.9];
mu{4} = [1.2; -0.4; -0.5];
Sigma{1} = 0.8*[0.45 0.10 0.05; 0.10 0.40 0.02; 0.05 0.02 0.35];
Sigma{2} = 0.8*[0.50 0.08 0.00; 0.08 0.35 -0.03; 0.00 -0.03 0.30];
Sigma{3} = 0.8*[0.40 0.05 0.02; 0.05 0.55 0.06; 0.02 0.06 0.45];
Sigma{4} = 0.8*[0.48 -0.06 0.00; -0.06 0.38 0.04; 0.00 0.04 0.32];
priors = ones(1,C)/C;

%% ---------------- Data generator & MAP ----------------
function [X, labels] = genDataGaussianMixture(N, mu, Sigma, priors)
    C = numel(mu); n = numel(mu{1});
    X = zeros(n, N); labels = zeros(1, N);
    edges = [0, cumsum(priors)]; u = rand(1, N);
    for c = 1:C
        idx = find(u >= edges(c) & u < edges(c+1));
        nc = numel(idx); if nc == 0, continue; end
        S = Sigma{c}; [R,p] = chol(S);
        if p > 0, S = S + 1e-8*eye(n); R = chol(S); end
        Z = randn(n, nc); X(:,idx) = R'*Z + mu{c}; labels(idx) = c;
    end
end

function pred = MAP_decision(X, mu, Sigma, priors)
    C = numel(mu); n = size(X,1); N = size(X,2);
    logp = -inf(C,N);
    for c = 1:C
        S = Sigma{c}; [R,p] = chol(S);
        if p > 0, S = S + 1e-8*eye(n); R = chol(S); end
        diff = X - mu{c};
        y = R'\diff;
        logp(c,:) = -0.5*(sum(y.^2,1) + 2*sum(log(diag(R))) + n*log(2*pi)) + log(priors(c));
    end
    [~, pred] = max(logp, [], 1);
end

%% ---------------- Bayes baseline ----------------
[Xtest, ytest] = genDataGaussianMixture(1e5, mu, Sigma, priors);
pred_bayes = MAP_decision(Xtest, mu, Sigma, priors);
bayes_error = mean(pred_bayes ~= ytest);
fprintf('Bayes (MAP) error = %.4f\n', bayes_error);

%% ---------------- Training setup ----------------
train_sizes = [100, 500, 1000, 5000, 10000];
P_candidates = [4,8,12,16,24,32,48,64,96,128];
options_base = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',500, ...
    'MiniBatchSize',128, ...
    'L2Regularization',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false);

useELU = exist('eluLayer','file')==2;
if useELU
    fprintf('Using built-in ELU activation layer.\n');
else
    fprintf('Using custom ISRU activation layer (alpha=0.5).\n');
end

%% ---------------- Helper: Train one MLP ----------------
function net = train_mlp(Xtrain, ytrain, P, opts, useELU)
    n = size(Xtrain,1);
    X = Xtrain'; Y = categorical(ytrain(:));
    if useELU
        act = eluLayer('Name','act1');
    else
        act = ISRULayer(0.5,'isru1');
    end
    layers = [
        featureInputLayer(n,'Normalization','zscore')
        fullyConnectedLayer(P,'WeightsInitializer','glorot')
        act
        fullyConnectedLayer(4)
        softmaxLayer
        classificationLayer];
    opts.MiniBatchSize = min(opts.MiniBatchSize, size(X,1));
    net = trainNetwork(X, Y, layers, opts);
end

%% ---------------- Cross-validation + Testing (Improved) ----------------
nCVRepeats = 3;
enforceMonotonicP = true;

results = table('Size',[numel(train_sizes),3], ...
    'VariableTypes',{'double','double','double'}, ...
    'VariableNames',{'Ntrain','BestP','MLP_test_error'});

for si = 1:numel(train_sizes)
    N = train_sizes(si);
    fprintf('\n=== Training size N = %d ===\n', N);

    [Xtr_all, ytr_all] = genDataGaussianMixture(N, mu, Sigma, priors);
    cv_errs = zeros(numel(P_candidates), nCVRepeats);

    for r = 1:nCVRepeats
        cv = cvpartition(ytr_all,'KFold',10);
        for pidx = 1:numel(P_candidates)
            P = P_candidates(pidx); fold_err = zeros(cv.NumTestSets,1);
            for f = 1:cv.NumTestSets
                tr = training(cv,f); vl = test(cv,f);
                net = train_mlp(Xtr_all(:,tr), ytr_all(tr), P, options_base, useELU);
                Ypred = classify(net, Xtr_all(:,vl)');
                fold_err(f) = mean(double(Ypred)' ~= ytr_all(vl));
            end
            cv_errs(pidx,r) = mean(fold_err);
        end
    end

    mean_cv_err = mean(cv_errs,2);
    std_cv_err = std(cv_errs,0,2);

    figure;
    errorbar(P_candidates, mean_cv_err, std_cv_err, '-o','LineWidth',1.5);
    xlabel('Hidden units P'); ylabel('Mean CV error');
    title(sprintf('CV error vs P for N=%d', N)); grid on;

    [~,bestIdx] = min(mean_cv_err);
    Pstar = P_candidates(bestIdx);
    if enforceMonotonicP && si>1
        Pstar = max(Pstar, results.BestP(si-1));
    end
    fprintf('  Selected P* = %d\n',Pstar);

    nRestarts = 5; bestTrainLL = -inf; bestNet = [];
    for r = 1:nRestarts
        net_r = train_mlp(Xtr_all, ytr_all, Pstar, options_base, useELU);
        probs = predict(net_r, Xtr_all');
        idx = sub2ind(size(probs), (1:size(probs,1))', ytr_all(:));
        loglik = sum(log(max(probs(idx),1e-12)));
        if loglik > bestTrainLL
            bestTrainLL = loglik; bestNet = net_r;
        end
    end

    Ypred = classify(bestNet, Xtest');
    test_err = mean(double(Ypred)' ~= ytest);

    results.Ntrain(si)=N;
    results.BestP(si)=Pstar;
    results.MLP_test_error(si)=test_err;
    fprintf('  Best restart train LL = %.2f, Test err = %.4f\n',bestTrainLL,test_err);
end

%% ---------------- Plot results ----------------
figure;
semilogx(results.Ntrain,results.MLP_test_error,'-o','LineWidth',1.8);
hold on; yline(bayes_error,'--r','LineWidth',1.5);
xlabel('Training samples (log scale)'); ylabel('Test P(error)');
legend('MLP','Bayes (optimal)','Location','best');
title('MLP test error vs Bayes error');
grid on;
disp(results);
