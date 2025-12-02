%% ============================================================
%  EECE5644 – Assignment 4 – Question 2
%  GMM Image Segmentation with K-fold CV Model Selection
% =============================================================

clearvars; close all; clc;
rng(1); % reproducibility

% -------------------------------------------------------------
% Input images (you may choose one image as assignment requires)
% -------------------------------------------------------------
filenames = { ...
    '3096_gray.jpg', ...
    '42049_gray.jpg', ...
    '3096_color.jpg', ...
    '42049_color.jpg' ...
};

% Candidate mixture sizes
Kvalues = [2 3 4];

% Cross-validation settings
numFolds     = 5;
maxPixelsCV  = 50000;  % pixel cap for CV
replicatesEM = 3;
covType      = 'full';

figure(1); clf;

for imageCounter = 1:numel(filenames)

    %-----------------------------------------------------
    % Load image
    % -----------------------------------------------------
    fname  = filenames{imageCounter};
    imdata = imread(fname);

    % Convert indexed images to RGB if needed
    if ndims(imdata)==2
        isGray = true;
    elseif ndims(imdata)==3
        isGray = false;
    else
        error('Unsupported image format.');
    end

    % Show original
    subplot(numel(filenames), length(Kvalues)+1, (imageCounter-1)*(length(Kvalues)+1) + 1);
    imshow(imdata); title('Original Image');

    % -----------------------------------------------------
    % Construct feature vectors (spatial + intensity or RGB)
    % -----------------------------------------------------
    if isGray
        % Grayscale: (row, col, intensity)
        [R,C] = size(imdata); 
        N = R*C;
        imdata = double(imdata);

        [rowGrid, colGrid] = ndgrid(1:R, 1:C);
        f1 = rowGrid(:)';
        f2 = colGrid(:)';
        f3 = imdata(:)';

        features = [f1; f2; f3]; % (3 x N)

    else
        % Color: (row, col, R, G, B)
        [R,C,D] = size(imdata); 
        imdata = double(imdata);
        N = R*C;

        [rowGrid, colGrid] = ndgrid(1:R, 1:C);

        f1 = rowGrid(:)';
        f2 = colGrid(:)';

        rgb = reshape(imdata, [], 3)'; % 3 x N

        features = [f1; f2; rgb]; % (5 x N)
    end

    d = size(features,1);

    % -----------------------------------------------------
    % Normalize each feature dimension to [0,1]
    % -----------------------------------------------------
    minf   = min(features,[],2);
    maxf   = max(features,[],2);
    ranges = maxf - minf;
    ranges(ranges==0) = 1;

    x = (features - minf) ./ ranges;  % (d x N), each ∈ [0,1]

    % -----------------------------------------------------
    % Subsample for CV (if needed) & build CV dataset
    % -----------------------------------------------------
    if N > maxPixelsCV
        idxCV = randperm(N, maxPixelsCV);
    else
        idxCV = 1:N;
    end

    Xcv = x(:, idxCV)';  % (nCV x d)
    nCV = size(Xcv,1);

    % Adjust folds if too many
    if numFolds >= nCV
        numFolds = max(2, floor(nCV/2));
        warning('Reduced numFolds to %d due to limited CV samples', numFolds);
    end

    cv = cvpartition(nCV, 'KFold', numFolds);

    avgValLL  = zeros(1,length(Kvalues));  % store CV scores
    models_CV = cell(1,length(Kvalues));   % store CV models

    % -----------------------------------------------------
    % CV-based model order selection
    % -----------------------------------------------------
    optionsEM = statset('MaxIter',500, 'TolFun',1e-6, 'Display','off');

    for k = 1:length(Kvalues)
        K = Kvalues(k);
        foldLL = zeros(1,numFolds);

        for fold = 1:numFolds
            trainIdx = training(cv, fold);
            testIdx  = test(cv, fold);

            Xtrain = Xcv(trainIdx,:);
            Xval   = Xcv(testIdx,:);

            % Fit GMM with EM
            try
                gm = fitgmdist(Xtrain, K, ...
                    'CovarianceType', covType, ...
                    'RegularizationValue', 1e-6, ...
                    'Options', optionsEM, ...
                    'Replicates', replicatesEM, ...
                    'Start', 'plus');
            catch
                % Retry with larger regularization if singular
                gm = fitgmdist(Xtrain, K, ...
                    'CovarianceType', covType, ...
                    'RegularizationValue', 1e-4, ...
                    'Options', optionsEM, ...
                    'Replicates', 1);
            end

            % Use log(pdf + eps) to avoid -inf
            p = pdf(gm, Xval);
            foldLL(fold) = sum(log(p + eps));
        end

        avgValLL(k) = mean(foldLL);

        % Fit one model on the entire CV subset (for visualization)
        models_CV{k} = fitgmdist(Xcv, K, ...
            'CovarianceType', covType, ...
            'RegularizationValue', 1e-6, ...
            'Options', optionsEM, ...
            'Replicates', replicatesEM, ...
            'Start','plus');

        fprintf('Image %d | K = %d | Avg CV log-likelihood = %.3f\n', ...
                imageCounter, K, avgValLL(k));
    end

    % -----------------------------------------------------
    % Select best K
    % -----------------------------------------------------
    [~, bestIdx] = max(avgValLL);
    bestK = Kvalues(bestIdx);
    fprintf('Image %d | Selected K = %d (best by CV)\n', imageCounter, bestK);

    % -----------------------------------------------------
    % FINAL MODEL: Refit best-K GMM on FULL IMAGE (all pixels)
    % -----------------------------------------------------
    Xfull = x';  % (N x d)

    try
        gmBest = fitgmdist(Xfull, bestK, ...
            'CovarianceType', covType, ...
            'RegularizationValue', 1e-6, ...
            'Options', optionsEM, ...
            'Replicates', replicatesEM, ...
            'Start','plus');
    catch
        gmBest = fitgmdist(Xfull, bestK, ...
            'CovarianceType', covType, ...
            'RegularizationValue', 1e-4, ...
            'Options', optionsEM, ...
            'Replicates', 1);
    end

    %% -----------------------------------------------------
    % SEGMENTATION for all K (using CV-fitted models)
    %% -----------------------------------------------------
    for k = 1:length(Kvalues)
        K = Kvalues(k);

        % Use CV-trained models for consistent comparison
        gmK = models_CV{k};

        resp = posterior(gmK, Xfull);  
        [~, labels] = max(resp,[],2);

        labelImg = reshape(labels, R, C);

        subplot(numel(filenames), length(Kvalues)+1, ...
            (imageCounter-1)*(length(Kvalues)+1) + 1 + k);

        % scale labels 1..K → 0..255
        segVis = uint8((double(labelImg)-1)*(255/(K-1)));
        imshow(segVis);

        if K == bestK
            title(sprintf('GMM K=%d (best)', K));
        else
            title(sprintf('GMM K=%d', K));
        end
    end

end
