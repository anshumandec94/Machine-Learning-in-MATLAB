load('optdigits_train.txt')
load('optdigits_test.txt')
Lval=[2,4,9];
kval=[1,3,5];
training_classes = optdigits_train(:,65);
training_features = optdigits_train(:,1:64);
test_instances = optdigits_test(:,1:64);
test_classes = optdigits_test(:,65);
princ_mat = myLDA(optdigits_train);

for x=1:size(Lval,2)
    reduced_evectors = princ_mat(:,1:Lval(x));
    projected_training = training_features(:,1:64)*reduced_evectors;
    projected_test = test_instances(:,1:64)*reduced_evectors;
    for y=1:size(kval,2)
        error_rate=myKNN(projected_training,projected_test,training_classes,test_classes,kval(y));
        sprintf('Error rate for L=%d and K=%d using LDA is %2.2f percent',Lval(x),kval(y),error_rate)
    end
    
end
dim2_vectors = princ_mat(:,1:2);
p2_training = training_features(:,1:64)*dim2_vectors;
p2_test = test_instances(:,1:64)*dim2_vectors;
figure
hold on
subplot(2,1,1);
gscatter(p2_training(:,1),p2_training(:,2),training_classes);
axis([-3 3 -4 4]);
title('Training data along principal components');
p2_training = [p2_training training_classes];
for i=0:9
    classi=p2_training(p2_training(:,3)==i,1:2);
    randind = randsample(1:length(classi),10);
    classi_print=classi(randind,:);
    text(classi_print(:,1),classi_print(:,2),string(i));
end
p2_test =[p2_test test_classes];
subplot(2,1,2);
gscatter(p2_test(:,1),p2_test(:,2),test_classes);
axis([-3 3 -4 4]);
title('Test data along principal components');
for i=0:9
    classi=p2_test(p2_test(:,3)==i,1:2);
    randind = randsample(1:length(classi),10);
    classi_print=classi(randind,:);
    text(classi_print(:,1),classi_print(:,2),string(i));
end
