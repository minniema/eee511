CatDC=load('Category DC');
dataDC=load('dataset_DC');
CatPTO=load('Category PTO');
dataPTO=load('dataset_PTO');

%% split into 70 % training and 30% test

trainDC=dataDC.DC(1:round(length(dataDC.DC)*.7),:);
trainCatDC=CatDC.Category(1:round(length(dataDC.DC)*.7),:);
testDC=dataDC.DC(round(length(dataDC.DC)*.7)+1:end,:);
testCatDC=CatDC.Category(round(length(dataDC.DC)*.7)+1:end,:);

trainPTO=dataPTO.PTO(1:round(length(dataPTO.PTO)*.7),:);
trainCatPTO=CatPTO.Category(1:round(length(dataPTO.PTO)*.7),:);
testPTO=dataPTO.PTO(round(length(dataPTO.PTO)*.7)+1:end,:);
testCatPTO=CatDC.Category(round(length(dataPTO.PTO)*.7)+1:end,:);


%% Training neural network for DC
% net1 = feedforwardnet(10);
% net1=train(net1,Tnorm',y');
% ynet=sim(net1,Tnorm');
% ynetn=sim(net1,Tnewnorm');
% perf1 = mse(net1,Tnorm',y');
% figure
% scatter(T,y)
% hold on
% scatter(T,ynet)
% hold on 
% scatter(Tnew,ynetn)
% title("Neural Network 10 Neurons, 1 Hidden Layer")
% xlabel("Scaled Time Index ")
% ylabel("Scaled Output ")
% legend('Original','Train Output','30 Point Estimation')
