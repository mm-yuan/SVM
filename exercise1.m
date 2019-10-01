load iris

% generate random indices
idx = randperm(size(X,1));

% create the training and validation sets
% using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

% tune the sig2 while fix gam
%
gam = 1; sig2list=[0.001,0.01, 0.1, 1, 5, 10, 25];

errlist=[];
rate=[];
for sig2=sig2list,
  
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
 
% Obtain the output of the trained classifier for validation set
estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
err = sum(estYval~=Yval);errlist=[errlist; err];
end
rate=(errlist/length(Yt))*100;

% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(sig2list), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclass'),


% tune the gam while fix sig2
%
sig2 = 0.1; gamlist=[0.001,0.1, 1, 5,10,100,1000];

errlist=[];
rate=[];
for gam=gamlist,
  
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
 
% Obtain the output of the trained classifier for validation set
estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
err = sum(estYval~=Yval);errlist=[errlist; err];
end
rate=(errlist/length(Yt))*100;

% make a plot of the misclassification rate wrt. gam
%
figure;
plot(log(gamlist), errlist, '*-'), 
xlabel('log(gam)'), ylabel('number of misclass'),

% Perform crossvalidation using 10 folds
% tune the sig2 while fix gam
%
gam = 1;
sig2list=[0.001,0.01, 0.1, 1, 5, 10, 25];

perfslist = [];

for sig2=sig2list,
  performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'},10,'misclass');
  
  perfslist=[perfslist; performance];
end

%
% make a plot of the perfs
%
figure;
plot(log(sig2list), perfslist, '*-'), 
xlabel('log(sig2list)'), ylabel('Performance'),


% tune the gam while fix sig2
sig2 = 0.1; gamlist=[0.1, 1, 5,10,100,1000];

perfslist = [];

for gam=gamlist,
  performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'},10,'misclass');
  
  perfslist=[perfslist; performance];
end

%
% make a plot of the perfs
%
figure;
plot(log(gamlist), perfslist, '*-'), 
xlabel('log(gam)'), ylabel('Performance'),

% Perform crossvalidation using Leave1out
% tune the sig2 while fix gam
%
gam = 1;
sig2list=[0.001,0.01, 0.1, 1, 5, 10, 25];

perfslist = [];

for sig2=sig2list,
  performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass');
  
  perfslist=[perfslist; performance];
end

%
% make a plot of the perfs
%
figure;
plot(log(sig2list), perfslist, '*-'), 
xlabel('log(sig2list)'), ylabel('Performance'),


% tune the gam while fix sig2
sig2 = 0.1; gamlist=[0.1, 1, 5,10,100,1000];

perfslist = [];

for gam=gamlist,
  performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass');
  
  perfslist=[perfslist; performance];
end

%
% make a plot of the perfs
%
figure;
plot(log(gamlist), perfslist, '*-'), 
xlabel('log(gam)'), ylabel('Performance'),


%% tunelssvm
model = {X,Y,'c',[],[],'RBF_kernel','csa'}; %ds
[gam,sig2,cost] = tunelssvm(model,'gridsearch','crossvalidatelssvm',{10,'misclass'}); %simplex

%% ROC
idx = randperm(size(X,1));
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

gam = 1; sig2=2;

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
[Ysim,Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
roc(Ylatent,Yval);


%% Linear Case

load breast;
%[coeff,score,latent,tsquared,explained]=pca(trainset);
V=[trainset labels_train]; %plot 4th and 24th features
class1 = V(:,31) == 1;
scatter(V(class1,4), V(class1,24), 'b','s');
hold on
scatter(V(~class1,4), V(~class1,24), 'r','*');
xlabel('X4');
ylabel('X24');

X  = trainset;
Y  = labels_train;
Xt = testset;
Yt = labels_test;

type='c'; 
model = {X,Y,type,[],[],'lin_kernel'};
gam = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});

[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n On Test Data: N of Misclassifications = %d, Error Rate = %.2f%%\n', err, err/length(Yt)*100)

performance = crossvalidate({X,Y,type,gam,[],'lin_kernel'}, 10,'misclass');
fprintf('\n On 10-Fold Crossvalidation: Error Rate = %.2f%%\n', performance*100);


roc(Zt, Yt)

%% RBF Method

type='c'; 
model = {X,Y,type,[],[],'RBF_kernel'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});


[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

[Yht, Zt2] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n On Test Data: N of Misclassifications = %d, Error Rate = %.2f%%\n', err, err/length(Yt)*100)

performance = crossvalidate({X,Y,type,gam,sig2,'RBF_kernel'}, 10,'misclass');
fprintf('\n On 10-Fold Crossvalidation: Error Rate = %.2f%%\n', performance*100);


roc(Zt2, Yt)

