function [data,labels_forDisplay,temps,listFeatures,lesduree,featuresNamesInteractx2fx] = ...
    load_data_irt(chemin, nomExp, useInteraction)
%[Xtrain,Ytrain,temps,listFeatures,lesduree,nbFeatInit] = load_data_irt(chemin,nomExp, useInteraction);
% Loads the ORION-AE datasets AFTER FEATURE EXTRACTION. 
%
% This code can be used with some features extracted as exposed in the 
% paper [1], from the datasets "ORION-AE" (https://doi.org/10.7910/DVN/FBRDU0). 
%
% Raw data are made of 5 campaigns (B,C,D,E,F), each of about 1.8 GB. You 
% should get one directory per campaign, 7 subfolders (except for "C", see 
% the documentation of raw data for more details). Then apply your feature
% extraction then load it with this function.
% 
% This function can be used for example with the file 
% "mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat" available on
% GITHUB. This file has been obtained from the aforementioned method using
% raw data from campaign "B". There are three sensors + velocimeter, only
% sensor 3 (micro200HF) is loaded here. 
% 
% INPUTS: 
%   * chemin is the path to the .mat feature matrices made of the following
%   fields (see aforementioned mesure_B_DESSERRAGE....mat file):
%       - P3: sensor name (micro200HF) (matrix)
%       - labels3: ground truth (vector)
%       - listFeatures: names of features (cells)
%       - lesduree: duration of each period (vector)
%   * nomExp is the name of the specific file to be loaded
%   * useInteraction: compute a design matrix by x2fx function from Matlab.
% 
% OUTPUTS:
%   * data: feature matrix
%   * labels_forDisplay: labels
%   * temps: time index of AE signals
%   * listFeatures: name of features
%   * lesduree: duration of each period 
%   * featuresNamesInteractx2fx: name of column in data
%
% Example 
% c = '/home/emmanuel.ramasso/Documents/DATA/IRT/;
% n = 'mesure_B_DESSERRAGE_TH2_db45_14_sqtwolog_30_80_1100.mat'
% [data,labels_forDisplay,temps,listFeatures,lesduree,featuresNamesInteractx2fx] = ...
%    load_data_irt(c, n, false);
% 
%
%   [1] Emmanuel Ramasso, Thierry Denoeux, Gael Chevallier, Clustering 
%   acoustic emission data stream with sequentially appearing clusters 
%   using mixture models, Mechanical Systems and Signal Processing, 2021.
%
%
% Emmanuel Ramasso, emmanuel.ramasso@femto-st.fr
% 25 april 2020
%

fprintf('\nLoading training data...');
c = chemin; 
load([c filesep nomExp],'P3','labels3','listFeatures','lesduree'); % Load data

% Select X and labels
X = P3;
labels = labels3;
labels_forDisplay = labels;
X = fillmissing(X,'nearest');

% take some columns apart
temps = X(:,5);
vibroStart = X(:,3);
X(:,1:8) = []; % keep only features
listFeatures(1:8) = [];
fprintf('\nOk.');
rng default                             % For reproducibility

% Whether or not we standardize the data before applying the algorithm or
% within it if possible
% standardizeBefore = true;

% Converts a matrix of predictors X to a design matrix D for regression analysis.
% Combine all features using "simple" operations. Here pairwise products of
% columns. As such we can create non linearity in variables.
% % useInteraction = true;

% For prefiltering of data to ensure consistency over time since those are time series data.
sizeMedianFilter = 31;

% take a value in [0,1] representing a part of the data that will be kept
% for validation (ie never seen by the model)
HoldOutRate = 0.0;

% Prefiltering of data to ensure consistency over time since
% those are time series data. Ideally we should do it for the training
% set and testing set separately. TO DO LATER.
fprintf('\nPrefiltering...');
X = movmedian(X,sizeMedianFilter);
fprintf('\nOk.');

% Selection of features: take all below.
descripteurs = 1:size(X,2);
data = X(:,descripteurs);

% compute design matrix - in that case the name of features has to be
% changed too.
featuresNamesInteractx2fx = [];
if useInteraction
    fprintf('\nData interaction...');
    
    %data = [data, log(data + double(data<=realmin))];
    
    % term constant = 1
    % term per column = 32
    % les interaction par paire = 32*(32-1)/2 = 496
    % les carrés = 32 si quadratic
    % total = 1+32*(31)/2+32 = 529
    % total = 1+32*(31)/2+32+32 = 561
    featuresNamesInteractx2fx = ['constant', listFeatures];
    for i=1:length(listFeatures)
        for j=i+1:length(listFeatures)
            featuresNamesInteractx2fx{end+1} = [listFeatures{i} 'ET' listFeatures{j}];
        end
    end
    if 0
        data = x2fx(data,'interaction');
        fprintf('\nOk..');
    else
        data = x2fx(data,'quadratic');
        for i=1:length(listFeatures)
            featuresNamesInteractx2fx{end+1} = ['SQ_' listFeatures{i}];
        end
        fprintf('\nOk..');
    end
    data(:,1) = [];
    %warning('We need here to compute the combination of names of features - TO BE DONE...')
end
% % 
% % % Standardization before or not
% % if standardizeBefore
% %     fprintf('\nStandardize features training data...');
% %     [data,mean_data_train,sigma_data_train] = zscore(data);   
% %     fprintf('\nOk.\n');
% % end

