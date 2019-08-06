test_d=optdigits_test;
classes=test_d(:,end);
%similar to 2b, we use the min_h on the test dataset to generate z-values
%and use PCA to get coefficients to calculate the 2 and 3 principal
%components. 
ztest=test_d*min_w';
ztest2=ztest*pc2;
ztest2=[ztest2 classes];
figure 
hold on
gscatter((ztest2(:,1)),(ztest2(:,2)),classes);

for i=0:9
    classi=ztest2(ztest2(:,3)==i,1:2);
    randind = randsample(1:length(classi),10);
    classi_print=classi(randind,:);
    text(classi_print(:,1),classi_print(:,2),string(i));
end
title("test data set projected 2d");
figure 


ztest3=ztest*pc3;
ztest3=[ztest3 classes];
scatter3((ztest3(:,1)),(ztest3(:,2)),(ztest3(:,3)),20,classes);
colormap(jet(10));
for i=0:9
    class3i=ztest3(ztest3(:,4)==i,1:3);
    randind = randsample(1:length(class3i),10);
    class3i_print=class3i(randind,:);
    text(class3i_print(:,1),class3i_print(:,2),class3i_print(:,3),(string(i)));
end
title("test data set projected 3d");