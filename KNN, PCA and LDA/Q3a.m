training = load('face_train_data_960.txt');
testing = load('face_test_data_960.txt');

total = [training(:,1:960) ; testing(:,1:960)];
training_classes = training(:,961);
testing_classes = testing(:,961);

[principal_components,k] = myPCA(total);
first5efaces = principal_components(:,1:5);
figure()
for i=1:5
    subplot(3,2,i);
    imagesc(reshape(first5efaces(1:end,i),32,30)')
end


