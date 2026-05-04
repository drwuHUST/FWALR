%% This program produces main multi-task results of the paper "Feature Weighting Improves Pool-Based Sequential Active Learning for Regression," 
%% available at https://arxiv.org/abs/2604.02019.
%%
%% AL for offline regression on EnergyEfficiency http://archive.ics.uci.edu/ml/datasets/energy+efficiency
%%
%% Compare the following 7 algorithms:
%% 1. Random
%% 2. GSx
%% 3. FW-GSx
%% 4. MT-iGS
%% 5. FW-MT-iGS
%% 6. MT-GALR
%% 7. FW-MT-GALR

%% Dongrui WU, drwu@hust.edu.cn

clc; clearvars; close all;
warning off all; rng('default');
nRepeat=100; % number of repeats to get statistically significant results
lambda=10; % RR parameter
numAlgs=7;
data=load('./data/EE-Cooling.mat');
XAll=zscore(data.data(:,1:end-1)); YAll=data.data(:,end);
numAll=length(YAll);
data=load('./data/EE-Heating.mat');
YAll=[YAll data.data(:,end)];
numTrainPool=round(numAll*.3);
minN=5;  % mininum number of training samples
maxN=40; % maximum number of training samples
RMSEs=nan(numAlgs,maxN,nRepeat,size(YAll,2)); CCs=RMSEs;
numSamples=nan(numAlgs,maxN,nRepeat);

%% Iterative approaches
for r=1:nRepeat
    r

    %% 30% training, 70% testing
    idsTrainPool=datasample(1:numAll,numTrainPool,'Replace',false);
    XTrainPool=XAll(idsTrainPool,:); YTrainPool=YAll(idsTrainPool,:);
    idsTrain=repmat(datasample(1:numTrainPool,maxN,'Replace',false),numAlgs,1);
    distX=squareform(pdist(XTrainPool));
    idsTest=1:numAll; idsTest(idsTrainPool)=[]; numTest=length(idsTest);
    XTest=XAll(idsTest,:); YTest=YAll(idsTest,:);
    b=zeros(size(XTrainPool,2)+1,numAlgs,3); % Store the regression coefficients of all models

    for n=minN:maxN
        YPred=zeros(size(XTest,1),numAlgs,2); % Store the predictions of all models

        %% 1. BL: Random sampling
        idxAlg=1;
        for k=1:size(YTrainPool,2)
            b(:,idxAlg,k)=ridge(YTrainPool(idsTrain(idxAlg,1:n),k),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg,k)=b(1,idxAlg,k)+XTest*b(2:end,idxAlg,k);
            RMSEs(idxAlg,n,r,k)=sqrt(mean((YPred(:,idxAlg,k)-YTest(:,k)).^2));
            CCs(idxAlg,n,r,k)=corr(YPred(:,idxAlg,k),YTest(:,k));
        end

        %% 2. GSx
        idxAlg=2;
        if n==minN
            [~,idsTrain(idxAlg,1)]=min(mean(distX,2));
            idsUnlabeled=1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1))=[];
            for i=2:n
                [~,idx]=max(min(distX(idsUnlabeled,idsTrain(idxAlg,1:i-1)),[],2));
                idsTrain(idxAlg,i)=idsUnlabeled(idx);
                idsUnlabeled(idx)=[];
            end
        end
        for k=1:size(YTrainPool,2)
            b(:,idxAlg,k)=ridge(YTrainPool(idsTrain(idxAlg,1:n),k),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg,k)=b(1,idxAlg,k)+XTest*b(2:end,idxAlg,k);
            RMSEs(idxAlg,n,r,k)=sqrt(mean((YPred(:,idxAlg,k)-YTest(:,k)).^2));
            CCs(idxAlg,n,r,k)=corr(YPred(:,idxAlg,k),YTest(:,k));
        end
        % Select a new sample by GSx
        if n<maxN
            idsUnlabeled=1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n))=[];
            [~,idx]=max(min(distX(idsUnlabeled,idsTrain(idxAlg,1:n)),[],2));
            idsTrain(idxAlg,n+1)=idsUnlabeled(idx);
        end

        %% 3. FW-MT-GSx
        idxAlg=3;
        if n==minN
            b(:,idxAlg,:)=b(:,2,:);  YPred(:,idxAlg,:)=YPred(:,2,:); 
            idsTrain(idxAlg,1:n,:)=idsTrain(2,1:n,:);
            RMSEs(idxAlg,n,r,:)=RMSEs(2,n,r,:); CCs(idxAlg,n,r,:)=CCs(2,n,r,:);
        else
            for k=1:size(YTrainPool,2)
                b(:,idxAlg,k)=ridge(YTrainPool(idsTrain(idxAlg,1:n),k),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg,k)=b(1,idxAlg,k)+XTest*b(2:end,idxAlg,k);
                RMSEs(idxAlg,n,r,k)=sqrt(mean((YPred(:,idxAlg,k)-YTest(:,k)).^2));
                CCs(idxAlg,n,r,k)=corr(YPred(:,idxAlg,k),YTest(:,k));
            end
        end
        % Select a new sample by FW-MT-GSx
        if n<maxN
            idsUnlabeled=1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n))=[];
            distX3=zeros(length(idsUnlabeled),n,size(YTrainPool,2));
            for k=1:size(YTrainPool,2)
                distX3(:,:,k)=pdist2(XTrainPool(idsUnlabeled,:).*b(2:end,idxAlg,k)',XTrainPool(idsTrain(idxAlg,1:n),:).*b(2:end,idxAlg,k)');
            end
            [~,idx]=max(min(prod(distX3,3),[],2));
            idsTrain(idxAlg,n+1)=idsUnlabeled(idx);
        end


        %% 4. MT-iGS
        idxAlg=4;
        if n==minN
            b(:,idxAlg,:)=b(:,2,:);  YPred(:,idxAlg,:)=YPred(:,2,:); 
            idsTrain(idxAlg,1:n,:)=idsTrain(2,1:n,:);
            RMSEs(idxAlg,n,r,:)=RMSEs(2,n,r,:); CCs(idxAlg,n,r,:)=CCs(2,n,r,:);
        else
            for k=1:size(YTrainPool,2)
                b(:,idxAlg,k)=ridge(YTrainPool(idsTrain(idxAlg,1:n),k),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg,k)=b(1,idxAlg,k)+XTest*b(2:end,idxAlg,k);
                RMSEs(idxAlg,n,r,k)=sqrt(mean((YPred(:,idxAlg,k)-YTest(:,k)).^2));
                CCs(idxAlg,n,r,k)=corr(YPred(:,idxAlg,k),YTest(:,k));
            end
        end
        % Select a new sample by MT-iGS
        if n<maxN
            idsUnlabeled=1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n))=[];
            distY3=zeros(numTrainPool-n,n,size(YTrainPool,2));
            for k=1:size(YTrainPool,2)
                YTrainUnlabeled=b(1,idxAlg,k)+XTrainPool(idsUnlabeled,:)*b(2:end,idxAlg,k); % Predictions for the num-Y unlabeled samples
                for i=1:n
                    distY3(:,i,k)=abs(YTrainUnlabeled-YTrainPool(idsTrain(idxAlg,i),k));
                end
            end
            [~,idx]=max(min(distX(idsUnlabeled,idsTrain(idxAlg,1:n)).*prod(distY3,3),[],2));
            idsTrain(idxAlg,n+1)=idsUnlabeled(idx);
        end

        %% 5. FW-MT-iGS
        idxAlg=5;
        if n==minN
            b(:,idxAlg,:)=b(:,4,:);  YPred(:,idxAlg,:)=YPred(:,4,:); 
            idsTrain(idxAlg,1:n,:)=idsTrain(4,1:n,:);
            RMSEs(idxAlg,n,r,:)=RMSEs(4,n,r,:); CCs(idxAlg,n,r,:)=CCs(4,n,r,:);
        else
            for k=1:size(YTrainPool,2)
                b(:,idxAlg,k)=ridge(YTrainPool(idsTrain(idxAlg,1:n),k),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg,k)=b(1,idxAlg,k)+XTest*b(2:end,idxAlg,k);
                RMSEs(idxAlg,n,r,k)=sqrt(mean((YPred(:,idxAlg,k)-YTest(:,k)).^2));
                CCs(idxAlg,n,r,k)=corr(YPred(:,idxAlg,k),YTest(:,k));
            end
        end
        % Select a new sample by FW-MT-iGS
        if n<maxN
            idsUnlabeled=1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n))=[];
            distX3=zeros(length(idsUnlabeled),n,size(YTrainPool,2)); distY3=distX3;
            for k=1:size(YTrainPool,2)
                distX3(:,:,k)=pdist2(XTrainPool(idsUnlabeled,:).*b(2:end,idxAlg,k)',XTrainPool(idsTrain(idxAlg,1:n),:).*b(2:end,idxAlg,k)');
                YTrainUnlabeled=b(1,idxAlg,k)+XTrainPool(idsUnlabeled,:)*b(2:end,idxAlg,k); % Predictions for the num-Y unlabeled samples
                for i=1:n
                    distY3(:,i,k)=abs(YTrainUnlabeled-YTrainPool(idsTrain(idxAlg,i),k));
                end
            end
            [~,idx]=max(min(prod(distX3,3).*prod(distY3,3),[],2));
            idsTrain(idxAlg,n+1)=idsUnlabeled(idx);
        end

        %% 6. GALR
        idxAlg=6;
        if n==minN
            b(:,idxAlg,:)=b(:,2,:);  YPred(:,idxAlg,:)=YPred(:,2,:);
            idsTrain(idxAlg,1:n,:)=idsTrain(2,1:n,:);
            RMSEs(idxAlg,n,r,:)=RMSEs(2,n,r,:); CCs(idxAlg,n,r,:)=CCs(2,n,r,:);
        end
        for k=1:size(YTrainPool,2)
            b(:,idxAlg,k)=ridge(YTrainPool(idsTrain(idxAlg,1:n),k),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
            YPred(:,idxAlg,k)=b(1,idxAlg,k)+XTest*b(2:end,idxAlg,k);
            RMSEs(idxAlg,n,r,k)=sqrt(mean((YPred(:,idxAlg,k)-YTest(:,k)).^2));
            CCs(idxAlg,n,r,k)=corr(YPred(:,idxAlg,k),YTest(:,k));
        end
        % Select a new sample by GALR
        if n<maxN
            idsUnlabeled=1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n))=[];
            [~,idx]=max(min(pdist2(XTrainPool(idsUnlabeled,:),XTrainPool(idsTrain(idxAlg,1:n),:),'cityblock'),[],2));
            idsTrain(idxAlg,n+1)=idsUnlabeled(idx);
        end

        %% 7. FW-MT-GALR
        idxAlg=7;
        if n==minN
            b(:,idxAlg,:)=b(:,2,:);  YPred(:,idxAlg,:)=YPred(:,2,:); 
            idsTrain(idxAlg,1:n,:)=idsTrain(2,1:n,:);
            RMSEs(idxAlg,n,r,:)=RMSEs(2,n,r,:); CCs(idxAlg,n,r,:)=CCs(2,n,r,:);
        else
            for k=1:size(YTrainPool,2)
                b(:,idxAlg,k)=ridge(YTrainPool(idsTrain(idxAlg,1:n),k),XTrainPool(idsTrain(idxAlg,1:n),:),lambda,0);
                YPred(:,idxAlg,k)=b(1,idxAlg,k)+XTest*b(2:end,idxAlg,k);
                RMSEs(idxAlg,n,r,k)=sqrt(mean((YPred(:,idxAlg,k)-YTest(:,k)).^2));
                CCs(idxAlg,n,r,k)=corr(YPred(:,idxAlg,k),YTest(:,k));
            end
        end
        % Select a new sample by FW-MT-GALR
        if n<maxN
            idsUnlabeled=1:numTrainPool; idsUnlabeled(idsTrain(idxAlg,1:n))=[];
            distX3=zeros(length(idsUnlabeled),n,size(YTrainPool,2));
            for k=1:size(YTrainPool,2)
                distX3(:,:,k)=pdist2(XTrainPool(idsUnlabeled,:).*b(2:end,idxAlg,k)',XTrainPool(idsTrain(idxAlg,1:n),:).*b(2:end,idxAlg,k)','cityblock');
            end
            [~,idx]=max(min(prod(distX3,3),[],2));
            idsTrain(idxAlg,n+1)=idsUnlabeled(idx);
        end

    end
end
save('EE_RR.mat','RMSEs','CCs','nRepeat','numAlgs','minN','maxN','numSamples');

%%
close all;
legendText={'BL','GSx','FW-GSx','MT-iGS','FW-MT-iGS','MT-GALR','FW-MT-GALR'};
linestyle={'k-','r-','r--','b-','b--','g-','g--','m-','m--'};
affect={'EE-Cooling','EE-Heating'};
load('EE_RR.mat');
mmMSEs=zeros(numAlgs,maxN); mmCCs=mmMSEs;
figure; set(gcf,'Position',[0 0 1800 900]);
set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',10);
for k=1:2
    mMSEs=squeeze(nanmean(RMSEs(:,:,:,k),3));
    mCCs=squeeze(nanmean(CCs(:,:,:,k),3));
    mmMSEs=mmMSEs+mMSEs/3; mmCCs=mmCCs+mCCs/3;
    subplot(3,4,k); hold on;
    for i=1:numAlgs
        plot(minN:maxN,mMSEs(i,minN:maxN),linestyle{i},'linewidth',1);
    end
    xlabel('$K$','fontsize',10,'interpreter','latex');     ylabel('RMSE','fontsize',10);
    axis tight; box on; title(affect{k},'fontsize',12,'interpreter','latex');
    h=legend(legendText); set(h,'fontsize',11)

    subplot(3,4,4+k); hold on;
    for i=1:numAlgs
        plot(minN:maxN,mCCs(i,minN:maxN),linestyle{i},'linewidth',1);
    end
    xlabel('$M$','fontsize',10,'interpreter','latex');     ylabel('CC','fontsize',10);
    axis tight; box on; title(affect{k},'fontsize',12,'interpreter','latex');
    h=legend(legendText,'location','southeast'); set(h,'fontsize',11)

end
subplot(3,4,3); hold on;
for i=1:numAlgs
    plot(minN:maxN,mmMSEs(i,minN:maxN),linestyle{i},'linewidth',1);
end
xlabel('$M$','fontsize',10,'interpreter','latex');     ylabel('RMSE','fontsize',10);
axis tight; box on; title('Average','fontsize',12,'interpreter','latex');
h=legend(legendText);  set(h,'fontsize',11)

subplot(3,4,7); hold on;
for i=1:numAlgs
    plot(minN:maxN,mmCCs(i,minN:maxN),linestyle{i},'linewidth',1);
end
xlabel('$M$','fontsize',10,'interpreter','latex');     ylabel('CC','fontsize',10);
axis tight; box on; title('Average','fontsize',12,'interpreter','latex');
h=legend(legendText,'location','southeast'); set(h,'fontsize',11);

