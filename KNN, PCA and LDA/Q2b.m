load('optdigits_train.txt')
load('optdigits_test.txt')
knn = [1 3 5 7];
training_classes = optdigits_train(:,65);
training_features = optdigits_train(:,1:64);
test_instances = optdigits_test(:,1:64);
test_classes = optdigits_test(:,65);
[princ_components,k] = myPCA(training_features);
sprintf('The first k components are %d',k);
projected_training_data = optdigits_train(:,1:64)*princ_components(:,1:k);
projected_testing_data = optdigits_test(:,1:64)*princ_components(:,1:k);

%plot along 2 principal components
princ_components_2 = princ_components(:,1:2);
pc2_training_data = optdigits_train(:,1:64)*princ_components_2;
pc2_test_data = optdigits_test(:,1:64)*princ_components_2;
for j=1:size(knn,2)
    error_rate = myKNN(projected_training_data,projected_testing_data,training_classes,test_classes,knn(j));
    sprintf('Error rate for k=%d with projected_data is %2.2f percent',knn(j),error_rate)
end
figure
hold on
subplot(2,1,1)
gscatter(pc2_training_data(:,1),pc2_training_data(:,2),training_classes);
axis([-35 35 -30 35]);
title('Training data along projected components');

%text(pc2_training_data(:,1),pc2_training_data(:,2),string(training_classes));

pc2_training_data = [pc2_training_data training_classes];
for i=0:9
    classi=pc2_training_data(pc2_training_data(:,3)==i,1:2);
    randind = randsample(1:length(classi),10);
    classi_print=classi(randind,:);
    text(classi_print(:,1),classi_print(:,2),string(i));
end
subplot(2,1,2)
gscatter(pc2_test_data(:,1),pc2_test_data(:,2),test_classes);
axis([-35 35 -30 35]);
pc2_test_data = [pc2_test_data test_classes];
title('Test data along projected components');
for i=0:9
    classi=pc2_test_data(pc2_test_data(:,3)==i,1:2);
    randind = randsample(1:length(classi),10);
    classi_print=classi(randind,:);
    text(classi_print(:,1),classi_print(:,2),string(i));
end

