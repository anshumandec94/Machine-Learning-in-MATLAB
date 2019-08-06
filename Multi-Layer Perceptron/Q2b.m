
train_d=optdigits_train;
valid_d=optdigits_valid;
combined=[train_d; valid_d];
[r,c]=size(combined);
%we use the previously generated min_h values to calculate the z-values
%for the combined dataset
z_comb=zeros(r,min_h);
classes=combined(:,end);
combined=[combined(:,1:end-1) ones(r,1)];
z_comb=combined*min_w';
%we can PCA for z_comb and use the coefficients to get our 2 and 3
%principal components
[principal_coeff]=pca(z_comb);
pc2=principal_coeff(:,1:2);
z2_data=z_comb*pc2;
z2_data=[z2_data classes];
figure
hold on
gscatter((z2_data(:,1)),(z2_data(:,2)),classes);

for i=0:9
    classi=z2_data(z2_data(:,3)==i,1:2);
    randind = randsample(1:length(classi),10);
    classi_print=classi(randind,:);
    text(classi_print(:,1),classi_print(:,2),string(i));
end
title("Combined data set projected 2d");


figure

pc3=principal_coeff(:,1:3);
z3_data=z_comb*pc3;
z3_data=[z3_data classes];

scatter3((z3_data(:,1)),(z3_data(:,2)),(z3_data(:,3)),20,classes);
colormap(jet(10));
for i=0:9
    class3i=z3_data(z3_data(:,4)==i,1:3);
    randind = randsample(1:length(class3i),10);
    class3i_print=class3i(randind,:);
    text(class3i_print(:,1),class3i_print(:,2),class3i_print(:,3),(string(i)));
end

title("Combined data set projected on 3d");