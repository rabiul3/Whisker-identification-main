clear all;
load('gamma_snippets.mat');
%========Preprocessing data to eliminate the missing or incomplete data=========
Y = feature_matrix;
labels =  [zeros(450,1); ones(450,1)]; % defined label in our case C2=0 and delta=1
% missing or incomplete data
incomplete_trials = [450 451];
bad_trials = [650 729 ];
Y([bad_trials  incomplete_trials],:) = [];
labels([bad_trials incomplete_trials]) = [];
% find bad time-channel samples
badsamples = find(mean(abs(Y))<=1e-3);
Y(:,badsamples) = [];
size(Y)
%================================================

%% Apply PCA to separate sources
noclasses = 2; % We have two class one is C2 and another is delta
[coeff, score, latent, tsquared, explained, mu] = pca(Y','NumComponents',noclasses-1);
%plot(coeff(:,1));
boxplot(coeff(:,1),labels)
xlabel('Group: C2 (0) and Delta (1)')
ylabel('coefficient values')
%%

%=========Classify based on kmeans to evaluate the PCA method=========
[idx,ctrs] = kmeans(coeff,noclasses,'Distance','cosine',...
    'Replicates',500);
idx = bestMap(labels,idx);
%============= evaluate AC: accuracy ==============
acc  = length(find(labels == idx))/length(labels);
%============= evaluate MIhat: nomalized mutual information =================
MIhat = MutualInfo2(labels,idx);
fprintf('Accuracy %.2f\n',acc)
fprintf('Mutual information %.2f\n',MIhat)