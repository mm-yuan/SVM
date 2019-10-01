
X = (-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);

Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

gam = 10;
sig2 = 0.001;
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});
YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
figure;
plot(Xtest,Ytest,'.');
hold on;
plot(Xtest,YtestEst,'r+');
legend('Ytest','YtestEst'),

cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10);
cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2});

optFun = 'simplex'; %% simplex
globalOptFun = 'csa';  %% ds
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel',globalOptFun},optFun,'crossvalidatelssvm',{10,'mse'})

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});


% function estimation
sig2 = 0.01; gam = 10;

criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1);
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2);
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3);

[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');



% Automatic Relevance Determination
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*(X(:,1))) + 0.3.*randn(100,1);
% optFun = 'simplex';
optFun = 'gridsearch';
globalOptFun = 'csa';
% globalOptFun = 'ds';

[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel',globalOptFun}, ...
    optFun, 'crossvalidatelssvm', {10, 'mse'});
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'},{alpha,b});

[selected, ranking] = bay_lssvmARD({X,Y,'class',gam,sig2});

time = (1:1:100)';
figure;
plot(time,Y);
plot(X(:,1),Y);
plot(X(:,3),Y);


% Classification

load iris;
gam_list = [0.1,1,10];
sig_list = [0.5, 1];

for gam = gam_list,
    for sig2 = sig_list,
        bay_modoutClass({X,Y,'c', gam,sig2}, 'figure');
    end
end

%%% robust

X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));

out = [15 17 19];
Y(out) = 0.7 + 0.3 * rand(size(out));
out = [41 44 46];
Y(out) = 1.5 + 0.2 * rand(size(out));

gam = 100;
sig2 = 0.01;

[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'});
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b});


model = initlssvm(X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess');
costFun = 'rcrossvalidatelssvm';
 wFun = 'whuber';
% wFun = 'whampel';
% wFun = 'wlogistic';
%wFun = 'wmyriad';
model = tunelssvm(model, 'simplex', costFun, {10, 'mae'}, wFun);
model = robustlssvm(model);
plotlssvm(model);

load santafe;

order_list = 20:1:80;
mselist=[];
for order = order_list,
Xu     = windowize(Z,1:order+1);
Xtra   = Xu(1:end-order,1:order);
Ytra   = Xu(1:end-order,end); 
Xs     = Z(end-order+1:end,1);

[gam,sig2] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10,'mae'});

[alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'});

prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'},Xs,200);
mse = immse(prediction,Ztest);mselist=[mselist; mse];
end
MSE=mselist';
figure;
set(plot(order_list,MSE),'LineWidth',2);

%figure;plot([prediction Ztest]);

