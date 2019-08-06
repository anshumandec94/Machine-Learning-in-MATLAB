training = load('face_train_data_960.txt');
testing = load('face_test_data_960.txt');
kvals = [1 3 5 7];
train_data = training(:,1:960);
train_classes = training(:,961);
test_data = testing(:,1:960);
test_classes = testing(:,961);
[principal_components,k] =myPCA(train_data);
sprintf('The k value that explains 90 pct of the variance is %d',k)
kprotesting = test_data*principal_components(:,1:k);
kprotraining = train_data*principal_components(:,1:k);
for i = 1:length(kvals)
    error_rate = myKNN(kprotraining,kprotesting,train_classes,test_classes,kvals(i));
    sprintf('Error rate for k=%d is %2.2f',kvals(i),error_rate)
end
