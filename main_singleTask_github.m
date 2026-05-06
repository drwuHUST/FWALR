%% This program produces main single-task results of the paper "Feature Weighting Improves Pool-Based Sequential Active Learning for Regression"
%%
%% ALR on 11 datasets; 80% training pool, 20% test
%%
%% Use ridge regression to estimate the feature weights, and ridge regression for linear regression
%%
%% Compare 11 approaches:
%% 1. BL
%% 2. EMCM
%% 3. QBC
%% 4. RD
%% 5. FW-RD
%% 6. GSx
%% 7. FW-GSx
%% 8. iGS
%% 9. FW-iGS
%% 10. GALR
%% 11. FW-GALR

clc; clearvars; close all; rng(0);

datasets = {'Yacht','autoMPG','NO2','PM10','Housing','CPS','EE-Cooling','Concrete','Airfoil','Wine-white','BikeSharing'};
nRepeat = 10;      % Number of repeats to get statistically significant results
lambda = .1;        % Ridge regression parameter
numAlgs = 11;       % Number of algorithms under comparison
RMSEs = cell(1,length(datasets)); CCs = RMSEs; % Store the main results

dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(data) fprintf('%d-%d ', data{1},data{2})); % Print progress of parfor

for d = 1:length(datasets)
    temp = load(['./data/' datasets{d} '.mat']);
    data = temp.data;    clearvars temp;

    XAll = single(data(:,1:end-1)); XAll = zscore(XAll); % Features of all data; Single precision to reduce memory usage
    YAll = single(data(:,end));                          % Labels of all data

    numAll = length(YAll);                  % Number of total samples
    minN = min(20,size(XAll,2)+1);          % Mininum number of training samples, selected by different approaches
    maxN = min(60,max(20,ceil(.1*numAll))); % Maximum number of training samples, selected by different approaches

    dRMSE = cell(nRepeat, 1); % RMSE for the current dataset
    dCC = cell(nRepeat, 1);   % CC for the current dataset

    parfor r = 1:nRepeat
        warning off all;
        dataDisp = {d r}; dqWorker.send(dataDisp); % Display parfor progress
        rRMSE = nan(numAlgs, maxN);                % RMSE for the current repeat
        rCC = nan(numAlgs, maxN);                  % CC for the current repeat

        %% 80% samples as the training pool, and 20% for test
        numTrainPool = round(numAll*.8);   % Number of samples in the training pool
        idsTrainPool = datasample(1:numAll,numTrainPool,'Replace',false);
        XTrainPool = XAll(idsTrainPool,:); % Features in the training pool
        YTrainPool = YAll(idsTrainPool);   % Labels in the training pool
        idsTest = 1:numAll; idsTest(idsTrainPool) = []; % Indices of the test samples
        XTest = XAll(idsTest,:);           % Feature of the test samples
        YTest = YAll(idsTest);             % True labels of the test samples

        idsTrain = repmat(datasample(1:numTrainPool,maxN,'Replace',false),numAlgs,1); % Initialize the indices of selected training samples for all algorithms to random
        distX2 = squareform(pdist(XTrainPool));             % Precompute pairwise training sample L2 distances for GSx
        distX1 = squareform(pdist(XTrainPool,'cityblock')); % Precompute pairwise training sample L1 distances for Graph
        b = zeros(size(XTrainPool,2)+1,numAlgs);            % Store the regression coefficients of all models
        nBoots = 5;                                         % Number of base learners in EMCM and QBC

        for n = minN:maxN
            YPred = zeros(size(XTest,1),numAlgs);           % Store the predictions of all models

            %% 1. BL: Random sampling
            idxAlg = 1;
            b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);

            %% 2. EMCM
            idxAlg = 2;
            b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            %% Select new samples by EMCM
            C = max(1,ceil(n*rand(nBoots,n)));
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            YPred0 = b(1,idxAlg)+XTrainPool(idsUnlabeled,:)*b(2:end,idxAlg); % Prediction by the best model
            YPreds = zeros(numTrainPool-n,nBoots);                           % Predictions by bootstrap
            for i = 1:nBoots
                bb = ridge(YTrainPool(idsTrain(idxAlg,C(i,:))),XTrainPool(idsTrain(idxAlg,C(i,:)),:),lambda,0);
                YPreds(:,i) = [ones(length(idsUnlabeled),1) XTrainPool(idsUnlabeled,:)]*bb;
            end
            EMCM = zeros(1,length(idsUnlabeled));
            for i = 1:length(idsUnlabeled)
                for j = 1:nBoots
                    EMCM(i) = EMCM(i)+norm((YPreds(i,j)-YPred0(i))*XTrainPool(idsUnlabeled(i),:));
                end
            end
            [~,idx] = max(EMCM);
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);

            %% 3. QBC; the first minN samples were obtained randomly
            idxAlg = 3;
            b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            %% Select new samples by QBC
            C = max(1,ceil(n*rand(nBoots,n)));
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            YPreds = zeros(numTrainPool-n,nBoots); % Predictions by bootstrap
            for i = 1:nBoots
                bb = ridge(YTrainPool(idsTrain(idxAlg,C(i,:))),XTrainPool(idsTrain(idxAlg,C(i,:)),:),lambda,0);
                YPreds(:,i) = [ones(length(idsUnlabeled),1) XTrainPool(idsUnlabeled,:)]*bb;
            end
            QBC = var(YPreds,0,2);
            [~,idx] = max(QBC);
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);

            %% 4. RD
            idxAlg = 4;
            if n == minN     % Select centers of minN clusters as initialization
                [idsCluster,~,~,dist2Centroid] = kmeans(XTrainPool,n,'Replicates',10); % dist2Centroid: numY*n
                idsSamplesInCluster = cell(1,n); % Store the indices of samples in the n clusters
                for i = 1:n
                    idsSamplesInCluster{i} = find(idsCluster == i);
                end
                for k = 1:n
                    [~,idx] = min(dist2Centroid(idsCluster == k,k));  % Idx is the index of sample closest to the centroid
                    idsTrain(idxAlg,k) = idsSamplesInCluster{k}(idx); % Select the closest sample to label
                end
            end
            b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by RD
            [idsCluster,~,~,dist2Centroid] = kmeans(XTrainPool,n+1,'Replicates',10); % Cluster X into n+1 clusters
            idsSamplesInCluster = cell(1,n+1); % Store the indices of samples in the n+1 clusters
            for i = 1:n+1
                idsSamplesInCluster{i} = find(idsCluster == i);
            end
            [~,ids2] = sort(cellfun(@length,idsSamplesInCluster),'descend'); % Sort the clusters from the lagest to the smallest
            idsSamplesInCluster = idsSamplesInCluster(ids2); dist2Centroid = dist2Centroid(:,ids2); % Update the quantities accordingly
            for k = 1:n+1
                if sum(ismember(idsTrain(idxAlg,1:n),idsSamplesInCluster{k})) == 0 % The k-th cluster does not contain a labeled sample
                    [~,idx] = min(dist2Centroid(idsSamplesInCluster{k},k));        % idx is the index of sample closest to the centroid
                    idsTrain(idxAlg,n+1) = idsSamplesInCluster{k}(idx);            % Select the sample to label
                    break;
                end
            end

            %% 5. FW-RD, using weighted features
            idxAlg = 5;
            if n == minN % When no samples have been labeled, there is no way to compute the feature weights, so use RD
                b(:,idxAlg) = b(:,idxAlg-1); YPred(:,idxAlg) = YPred(:,idxAlg-1); idsTrain(idxAlg,1:n) = idsTrain(idxAlg-1,1:n);
            else         % Update the ridge regression model
                b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            end
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by FW-RD
            [idsCluster,~,~,dist2Centroid] = kmeans(XTrainPool.*b(2:end,idxAlg)',n+1,'Replicates',10); % Weight the features in clustering
            idsSamplesInClusterW = cell(1,n+1); % Store the indices of samples in the n+1 clusters
            for i = 1:n+1
                idsSamplesInClusterW{i} = find(idsCluster == i); % Indices of samples in the i-th cluster
            end
            [~,ids3] = sort(cellfun(@length,idsSamplesInClusterW),'descend'); % Sort the clusters from the lagest to the smallest
            idsSamplesInClusterW = idsSamplesInClusterW(ids3); dist2Centroid = dist2Centroid(:,ids3);
            for k = 1:n+1
                if sum(ismember(idsTrain(idxAlg,1:n),idsSamplesInClusterW{k})) == 0 % The k-th cluster does not contain a labeled sample
                    [~,idx] = min(dist2Centroid(idsSamplesInClusterW{k},k));        % idx is the index of sample closest to the centroid
                    idsTrain(idxAlg,n+1) = idsSamplesInClusterW{k}(idx);            % Select the sample to label
                    break;
                end
            end

            %% 6. GSx
            idxAlg = 6;
            if n == minN
                mDistX = mean(distX2,2); % mean distance to all samples
                [~,idsTrain(idxAlg,1)] = min(mDistX); % Select the first sample as the center one
                idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1)) = [];
                for i = 2:n
                    [~,idx] = max(min(distX2(idsUnlabeled,idsTrain(idxAlg,1:i-1)),[],2)); % min distance to all labeled samples
                    idsTrain(idxAlg,i) = idsUnlabeled(idx); idsUnlabeled(idx) = [];
                end
            end
            b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by GSx
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            [~,idx] = max(min(distX2(idsUnlabeled,idsTrain(idxAlg,1:n)),[],2));
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);

            %% 7. FW-GSx
            idxAlg = 7;
            if n == minN
                b(:,idxAlg) = b(:,idxAlg-1); YPred(:,idxAlg) = YPred(:,idxAlg-1); idsTrain(idxAlg,1:n) = idsTrain(idxAlg-1,1:n);
            else
                b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            end
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by FW-GSx
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            [~,idx] = max(min(pdist2(XTrainPool(idsUnlabeled,:).*b(2:end,idxAlg)',XTrainPool(idsTrain(idxAlg,1:n),:).*b(2:end,idxAlg)'),[],2));
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);

            %% 8. iGS
            idxAlg = 8;
            if n == minN % use GSx
                b(:,idxAlg) = b(:,idxAlg-2); YPred(:,idxAlg) = YPred(:,idxAlg-2); idsTrain(idxAlg,1:n) = idsTrain(idxAlg-2,1:n);
            else
                b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            end
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by iGS
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            distY = zeros(numTrainPool-n,n); % Store the distances from the numY-n unlabelded samples to the n labeled samples
            YTrainUnlabeled = b(1,idxAlg)+XTrainPool(idsUnlabeled,:)*b(2:end,idxAlg); % Predictions for the num-Y unlabeled samples
            for i = 1:n
                distY(:,i) = abs(YTrainUnlabeled-YTrainPool(idsTrain(idxAlg,i)));     % Distances from the numY-n unlabelded samples to the i-th labeled sample
            end
            [~,idx] = max(min(pdist2(XTrainPool(idsUnlabeled,:),XTrainPool(idsTrain(idxAlg,1:n),:)).*distY,[],2));
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);

            %% 9. FW-iGS
            idxAlg = 9;
            if n == minN
                b(:,idxAlg) = b(:,idxAlg-1); YPred(:,idxAlg) = YPred(:,idxAlg-1); idsTrain(idxAlg,1:n) = idsTrain(idxAlg-1,1:n);
            else
                b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            end
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by FW-iGS
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            distY = zeros(numTrainPool-n,n); % Store the distances from the numY-n unlabelded samples to the n labeled samples
            YTrainUnlabeled = b(1,idxAlg)+XTrainPool(idsUnlabeled,:)*b(2:end,idxAlg); % Predictions for the num-Y unlabeled samples
            for i = 1:n
                distY(:,i) = abs(YTrainUnlabeled-YTrainPool(idsTrain(idxAlg,i)));     % Distances from the numY-n unlabelded samples to the i-th labeled sample
            end
            [~,idx] = max(min(pdist2(XTrainPool(idsUnlabeled,:).*b(2:end,idxAlg)',XTrainPool(idsTrain(idxAlg,1:n),:).*b(2:end,idxAlg)').*distY,[],2));
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);

            %% 10. GALR
            idxAlg = 10;
            if n == minN % use GSx
                b(:,idxAlg) = b(:,idxAlg-4); YPred(:,idxAlg) = YPred(:,idxAlg-4); idsTrain(idxAlg,1:n) = idsTrain(idxAlg-4,1:n);
            else
                b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            end
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by Graph
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            [~,idx] = max(min(distX1(idsUnlabeled,idsTrain(idxAlg,1:n)),[],2));
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);

            %% 11. FW-GALR
            idxAlg = 11;
            if n == minN
                b(:,idxAlg) = b(:,idxAlg-1); YPred(:,idxAlg) = YPred(:,idxAlg-1); idsTrain(idxAlg,1:n) = idsTrain(idxAlg-1,1:n);
            else
                b(:,idxAlg) = ridge(YTrainPool(idsTrain(idxAlg,1:n)),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg) = b(1,idxAlg)+XTest*b(2:end,idxAlg);
            end
            rRMSE(idxAlg,n) = sqrt(mean((YPred(:,idxAlg)-YTest).^2));
            rCC(idxAlg,n) = corr(YPred(:,idxAlg),YTest);
            % Select a new sample by FW-Graph
            idsUnlabeled = 1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n)) = [];
            [~,idx] = max(min(pdist2(XTrainPool(idsUnlabeled,:).*b(2:end,idxAlg)',XTrainPool(idsTrain(idxAlg,1:n),:).*b(2:end,idxAlg)','cityblock'),[],2));
            idsTrain(idxAlg,n+1) = idsUnlabeled(idx);
        end

        dRMSE{r} = rRMSE;
        dCC{r} = rCC;
    end

    RMSEs{d} = nan(numAlgs, maxN, nRepeat);
    CCs{d} = nan(numAlgs, maxN, nRepeat);
    for r = 1:nRepeat
        RMSEs{d}(:,:,r) = dRMSE{r};
        CCs{d}(:,:,r) = dCC{r};
    end

    %% Plot results for the current dataset
    mRMSEs = squeeze(nanmean(RMSEs{d},3));
    mCCs = squeeze(nanmean(CCs{d},3));
    figure; set(gcf, 'Position', get(0, 'Screensize'));
    linestyle = {'k-','m-','c-','r-','r--','g-','g--','b-','b--','y-','y--','k:'};
    subplot(121); hold on;
    for i = 1:numAlgs
        semilogy(minN:maxN,mRMSEs(i,minN:maxN),linestyle{i},'linewidth',2); % RMSE
    end
    legend('BL','EMCM','QBC','RD','FW-RD','GSx','FW-GSx','iGS','FW-iGS','Graph','FW-Graph','location','eastoutside');
    axis tight;

    subplot(122); hold on;
    for i = 1:numAlgs
        plot(minN:maxN,mCCs(i,minN:maxN),linestyle{i},'linewidth',2); % CC
    end
    legend('BL','EMCM','QBC','RD','FW-RD','GSx','FW-GSx','iGS','FW-iGS','Graph','FW-Graph','location','eastoutside');
    axis tight; drawnow;
end
