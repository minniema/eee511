CatDC=load('Category DC');
dataDC=load('dataset_DC');
CatPTO=load('Category PTO');
dataPTO=load('dataset_PTO');

%% split into 70 % training and 30% test

trainDC=dataDC.DC(1:round(length(dataDC.DC)*.8),:);
trainCatDC=CatDC.Category(1:round(length(dataDC.DC)*.8),:);
testDC=dataDC.DC(round(length(dataDC.DC)*.8)+1:end,:);
testCatDC=CatDC.Category(round(length(dataDC.DC)*.8)+1:end,:);

trainPTO=dataPTO.PTO(1:round(length(dataPTO.PTO)*.8),:);
trainCatPTO=CatPTO.Category(1:round(length(dataPTO.PTO)*.8),:);
testPTO=dataPTO.PTO(round(length(dataPTO.PTO)*.8)+1:end,:);
testCatPTO=CatDC.Category(round(length(dataPTO.PTO)*.8)+1:end,:);


%% scaling
maxDC=max(max(dataDC.DC));
trainDC=trainDC/maxDC;
testDC=testDC/maxDC;

maxPTO=max(max(dataPTO.PTO));
trainPTO=trainPTO/maxPTO;
testPTO=testPTO/maxPTO;





%% Training neural network for DC
hiddensize=[10 10];
net1 = feedforwardnet(hiddensize);
net1=train(net1,trainDC',trainCatDC');
%ynet=sim(net1,);
ynetn=sim(net1,testDC');
ynetn(ynetn>0.7)=1;
ynetn(ynetn<=0.7)=0;
perf1 = mse(net1,testDC',testCatDC');

%testrounded=round(ynetn);

%confusion matrix generaiton for nn
confusionsNN1 = confusionmat(testCatDC, ynetn);
tp1=confusionsNN1(1, 1);
fp1=confusionsNN1(1, 2);
fn1=confusionsNN1(2, 1);
tn1=confusionsNN1(2, 2);
%calculating Prec,Recall,F scores
prec1= tp1 / (tp1 + fp1);
rec1= tp1 / (tp1 + fn1);
f1 = (2 * prec1 * rec1) / (prec1 + rec1);
%calculating roc for curve, positive class define as 1 which is eating
[X, Y, ~, AUC] = perfcurve(testCatDC, ynetn, '1');
figure
plot(X,Y)
    xlabel('False positive rate') 
    ylabel('True positive rate')
title('ROC for Classification NN for DC')

%% train neural network for PTO
hiddensize=[10 10];
net2 = feedforwardnet(hiddensize);
net2=train(net2,trainPTO',trainCatPTO');
%ynet=sim(net1,);
ynetn2=sim(net2,testPTO');
% select 0.5 as threshold
ynetn2(ynetn2>0.6)=1;
ynetn2(ynetn2<=0.6)=0;

perf2 = mse(net2,testPTO',testCatPTO');


confusionsNN2 = confusionmat(testCatPTO, ynetn2);
tp2=confusionsNN2(1, 1);
fp2=confusionsNN2(1, 2);
fn2=confusionsNN2(2, 1);
tn2=confusionsNN2(2, 2);
%calculating Prec,Recall,F scores
prec2= tp2 / (tp2 + fp2);
rec2= tp2 / (tp2 + fn2);
f2 = (2 * prec2 * rec2) / (prec2 + rec2);
%calculating roc for curve, positive class define as 1 which is eating
[X, Y, ~, bayesAUC] = perfcurve(testCatPTO, ynetn2, '1');
figure
plot(X,Y)
    xlabel('False positive rate') 
    ylabel('True positive rate')
title('ROC for Classification NN for PTO')
% figure
% scatter(,y)
% hold on
% scatter(T,ynet)
% hold on 
% scatter(Tnew,ynetn)
% title("Neural Network 10 Neurons, 1 Hidden Layer")
% xlabel("Scaled Time Index ")
% ylabel("Scaled Output ")
% legend('Original','Train Output','30 Point Estimation')
% 
% %% train SVM1 for DC
% svm= fitcsvm(trainDC, trainCatDC);
% [DCsvm, svmscore] =  predict(svm, testDC);
% perfsvm1=immse(DCsvm, testCatDC);
% 
% confusionsvm = confusionmat(testCatDC, DCsvm);
% tp3=confusionsvm(1, 1);
% fp3=confusionsvm(1, 2);
% fn3=confusionsvm(2, 1);
% tn3=confusionsvm(2, 2);
% %calculating Prec,Recall,F scores
% prec3= tp3 / (tp3 + fp3);
% rec3= tp3 / (tp3 + fn3);
% f3 = (2 * prec3 * rec3) / (prec3 + rec3);
% %calculating roc for curve, positive class define as 1 which is eating
% [X, Y, ~, svmAUC] = perfcurve(testCatDC, svmscore(:, 2), '1');
% figure
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by DC SVM')
% 
% %% train SVM for PTO
% svm2= fitcsvm(trainPTO, trainCatPTO);
% [ptosvm, svmscore2] =  predict(svm2, testPTO);
% perfsvm2=immse(ptosvm, testCatPTO);
% 
% confusionsvm2 = confusionmat(testCatPTO, PTOsvm);
% tp2=confusionsvm2(1, 1);
% fp2=confusionsvm2(1, 2);
% fn2=confusionsvm2(2, 1);
% tn2=confusionsvm2(2, 2);
% %calculating Prec,Recall,F scores
% prec2= tp2 / (tp2 + fp2);
% rec2= tp2 / (tp2 + fn2);
% f2 = (2 * prec2 * rec2) / (prec2 + rec2);
% %calculating roc for curve, positive class define as 1 which is eating
% [X, Y, ~, svmAUC] = perfcurve(testCatDC, svmscore(:, 2), '1');
% figure
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by DC SVM')
% 
% 
% %% train using gaussian kernal
% 
% svm3= fitcsvm(trainDC, trainCatDC,'KernelFunction','rbf',...
%     'BoxConstraint',Inf);
% [DCsvm3, svmscore3] =  predict(svm3, testDC);
% perfsvm3=immse(DCsvm3, testCatDC);
% 
% svm4= fitcsvm(trainPTO, trainCatPTO,'KernelFunction','rbf',...
% 'BoxConstraint',Inf);
% [PTOsvm4, svmscore4] =  predict(svm4, testPTO);
% perfsvm4=immse(PTOsvm4, testCatPTO);

