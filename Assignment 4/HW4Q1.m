
%% EECE5644 – Assignment 4 – Question 1
%% ============================================================
% 1) DATA GENERATION 
%% ============================================================
generateData = @(N) hw4data_gen(N);

N_train = 1000;   
N_test = 10000;   

[Xtrain, label_train] = generateData(N_train);
[Xtest , label_test ] = generateData(N_test);

%% ------------------------------------------
% Plot the data
figure; hold on; axis equal; grid on; set(gcf,'Color','w');
scatter(Xtrain(label_train==-1,1), Xtrain(label_train==-1,2), 10, [0 0.447 0.741],'filled');
scatter(Xtrain(label_train==1 ,1), Xtrain(label_train==1 ,2), 10, [0.85 0.325 0.098],'filled');
title('Training Data (from hw4data.m)');
xlabel('x_1'); ylabel('x_2');
legend('Class -1','Class +1');
xmin=min(Xtrain(:,1))-1; xmax=max(Xtrain(:,1))+1;
ymin=min(Xtrain(:,2))-1; ymax=max(Xtrain(:,2))+1;


%% ============================================================
% 2) SVM (Gaussian kernel) with 5-fold CV
%% ============================================================
% MODIFIED: Extended range to include smaller kernel scales for more wiggly boundaries
kernelScales = logspace(-2,1.5,10);  % Changed from logspace(-1,1.2,8)clear
boxConstraints = logspace(-2,2,10);   % More granular search
K = 5;
cvErr = nan(numel(kernelScales),numel(boxConstraints));

for i=1:numel(kernelScales)
    for j=1:numel(boxConstraints)
        
        mdl = fitcsvm(Xtrain, label_train', ...
            'KernelFunction','rbf', ...
            'KernelScale',kernelScales(i), ...
            'BoxConstraint',boxConstraints(j), ...
            'Standardize',true);
        
        mdlCV = crossval(mdl,'KFold',K);
        cvErr(i,j) = kfoldLoss(mdlCV);
    end
end

[minErr,idx] = min(cvErr(:));
[iBest,jBest] = ind2sub(size(cvErr), idx);
bestKS = kernelScales(iBest);
bestC = boxConstraints(jBest);

fprintf('SVM: Best KernelScale = %.4f, Best BoxConstraint = %.4f, CV Error = %.4f\n', bestKS, bestC, minErr);

svmBest = fitcsvm(Xtrain,label_train', ...
    'KernelFunction','rbf','KernelScale',bestKS, ...
    'BoxConstraint',bestC,'Standardize',true);

YhatSVM = predict(svmBest,Xtest);
Pe_svm = mean(YhatSVM(:) ~= label_test(:));


%% ============================================================
% 3) MLP (NN) with QUADRATIC ACTIVATION + 5-fold CV
%% ============================================================

XtrainQ = Xtrain;
XtestQ = Xtest;

Ytrain_soft = double(label_train==1)+1; 
Ytest_soft = double(label_test==1)+1;

Hvals = [8 12 16 20];
cvErrNN = zeros(numel(Hvals),1);
idxCV = crossvalind('Kfold', N_train, K);

options = trainingOptions('adam','MaxEpochs',25,'MiniBatchSize',128, ...
    'Verbose',false,'Shuffle','every-epoch');

for h=1:numel(Hvals)
    H = Hvals(h);
    foldErr = zeros(K,1);
    
    for k=1:K
        tr = (idxCV~=k);
        vl = (idxCV==k);
        
      
        layers = [
            featureInputLayer(size(XtrainQ,2))
            fullyConnectedLayer(H)
            functionLayer(@(X) X.^2, 'Name', 'quadratic')  % QUADRATIC ACTIVATION
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer
        ];
        
        net = trainNetwork(XtrainQ(tr,:), categorical(Ytrain_soft(tr)), layers, options);
        
        Yvl_pred = classify(net, XtrainQ(vl,:));
        Yvl_true = categorical(Ytrain_soft(vl));
        
        % FORCE COLUMN VECTORS (fix all dimension errors)
        Yvl_pred = Yvl_pred(:);
        Yvl_true = Yvl_true(:);
        
        foldErr(k) = mean(Yvl_pred ~= Yvl_true);
    end
    
    cvErrNN(h) = mean(foldErr);
end

[~,bestH_idx] = min(cvErrNN);
Hbest = Hvals(bestH_idx);

fprintf('NN: Best Hidden Units = %d, CV Error = %.4f\n', Hbest, cvErrNN(bestH_idx));

%% Final NN training
layersBest = [
    featureInputLayer(size(XtrainQ,2))
    fullyConnectedLayer(Hbest)
    functionLayer(@(X) X.^2, 'Name', 'quadratic')  % QUADRATIC ACTIVATION
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

netBest = trainNetwork(XtrainQ, categorical(Ytrain_soft), layersBest, options);

YhatNN = classify(netBest,XtestQ);
Pe_nn = mean(YhatNN(:) ~= categorical(Ytest_soft(:)));


%% ============================================================
% 4) DECISION BOUNDARY PLOTS (Test Set Performance)
%% ============================================================

[x1g,x2g] = meshgrid(linspace(xmin,xmax,350),linspace(ymin,ymax,350));
Xg = [x1g(:),x2g(:)];
XgQ = Xg;  

probsNN = predict(netBest,XgQ);
pPos = reshape(probsNN(:,2), size(x1g));

[~,score] = predict(svmBest,Xg);
score2 = reshape(score(:,2), size(x1g));

nn_corr = (YhatNN(:) == categorical(Ytest_soft(:)));
svm_corr = (YhatSVM(:) == label_test(:));


figure('Color',[0.95 0.95 0.95],'Position',[100 80 1400 600]);

subplot(2,1,1); hold on; box on;
contour(x1g,x2g,pPos,[0.5 0.5],'k--','LineWidth',1.7);
plot(Xtest(nn_corr,1), Xtest(nn_corr,2), '+','Color',[0 0.7 0],'MarkerSize',5);
plot(Xtest(~nn_corr,1),Xtest(~nn_corr,2),'o','Color',[0.8 0 0],'MarkerSize',4);
title(sprintf('Neural Network (Quadratic Activation) Performance (Error = %.2f%%)',100*Pe_nn));
xlabel('x_1'); ylabel('x_2');

subplot(2,1,2); hold on; box on;
contour(x1g,x2g,score2,[0 0],'k--','LineWidth',1.7);
plot(Xtest(svm_corr,1), Xtest(svm_corr,2), '+','Color',[0 0.7 0],'MarkerSize',5);
plot(Xtest(~svm_corr,1),Xtest(~svm_corr,2),'o','Color',[0.8 0 0],'MarkerSize',4);
title(sprintf('SVM (RBF Kernel) Performance (Error = %.2f%%)',100*Pe_svm));
xlabel('x_1'); ylabel('x_2');


%% ============================================================
% PLOT 1: SVM Cross-Validation Error Heatmap
%% ============================================================

figure('Color','w');
imagesc(log10(kernelScales), log10(boxConstraints), cvErr'); 
set(gca,'YDir','normal');

xlabel('log_{10}(KernelScale)');
ylabel('log_{10}(BoxConstraint)');
title('SVM Cross-Validation Error (Classification Error)');
colorbar;

hold on;
plot(log10(bestKS), log10(bestC), 'rx', 'MarkerSize', 15, 'LineWidth', 3);
text(log10(bestKS), log10(bestC), ' Best', 'Color','r','FontSize',12);

%% ============================================================
% PLOT 2: MLP CV Error vs Hidden Layer Size
%% ============================================================

figure('Color','w');
plot(Hvals, cvErrNN, 'o--','LineWidth',2,'MarkerSize',8);
hold on;
plot(Hbest, cvErrNN(bestH_idx), 'rs', 'MarkerSize',12,'LineWidth',2);

xlabel('Hidden Layer Size H');
ylabel('Cross-Validation Error');
title('MLP (Quadratic Activation) Cross-Validation Error vs. Hidden Layer Size');
grid on;

%% ============================================================
% PLOT 3: SVM DECISION BOUNDARY ON TRAINING SET
%% ============================================================

figure; hold on; axis equal; grid on; set(gcf,'Color','w');

% Plot training points
scatter(Xtrain(label_train==-1,1), Xtrain(label_train==-1,2), 12, [0 0.447 0.741], 'filled');
scatter(Xtrain(label_train==1 ,1), Xtrain(label_train==1 ,2), 12, [0.8500 0.3250 0.0980], 'filled');

% Grid for boundary
[x1g, x2g] = meshgrid(linspace(min(Xtrain(:,1))-1, max(Xtrain(:,1))+1, 350), ...
    linspace(min(Xtrain(:,2))-1, max(Xtrain(:,2))+1, 350));
Xg = [x1g(:), x2g(:)];

% SVM output
[~,score] = predict(svmBest, Xg);
score2 = reshape(score(:,2), size(x1g));

% Decision boundary (score = 0)
contour(x1g, x2g, score2, [0 0], 'k--', 'LineWidth', 2);

title(sprintf('SVM Decision Boundary on Training Set (C=%.3g, KS=%.3g)', bestC, bestKS));
xlabel('x_1'); ylabel('x_2');
legend('Class -1','Class +1','SVM Boundary');

%% ============================================================
% PLOT 4: NN DECISION BOUNDARY ON TRAINING SET
%% ============================================================

figure; hold on; axis equal; grid on; set(gcf,'Color','w');

% Plot training points
scatter(Xtrain(label_train==-1,1), Xtrain(label_train==-1,2), 12, [0 0.447 0.741], 'filled');
scatter(Xtrain(label_train==1 ,1), Xtrain(label_train==1 ,2), 12, [0.8500 0.3250 0.0980], 'filled');

% Create grid for boundary
[x1g, x2g] = meshgrid(linspace(min(Xtrain(:,1))-1, max(Xtrain(:,1))+1, 350), ...
    linspace(min(Xtrain(:,2))-1, max(Xtrain(:,2))+1, 350));
Xg = [x1g(:), x2g(:)];

% FIXED: No quadratic expansion
XgQ = Xg;

% Predict NN posterior on grid
probs = predict(netBest, XgQ);
pPos = reshape(probs(:,2), size(x1g)); % probability class +1

% NN boundary at p=0.5
contour(x1g, x2g, pPos, [0.5 0.5], 'k--', 'LineWidth', 2);

title(sprintf('Neural Network (Quadratic Activation) Boundary on Training Set (H = %d)', Hbest));
xlabel('x_1'); ylabel('x_2');
legend('Class -1','Class +1','NN Boundary');


%% ============================================================
% INTERNAL FUNCTION
%% ============================================================
function [data_x, label] = hw4data_gen(N)

r_neg = 2;
r_pos = 4;
sig = 1;

noise = sig * randn(N,2);
prior = 0.5;

label = zeros(1,N);
u = rand(1,N);
label(u <= prior) = 1;
label(label==0) = -1;

theta = unifrnd(-pi,pi,N,1);

r_vec = zeros(N,1);
r_vec(label==1) = r_pos;
r_vec(label==-1) = r_neg;

data_x = r_vec .* [cos(theta), sin(theta)];
data_x = data_x + noise;

end