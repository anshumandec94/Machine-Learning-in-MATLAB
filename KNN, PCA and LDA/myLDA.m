function [projection_matrix] = myLDA(training_data)

class_group_data = {};
mean_data = [];
for i=0:9
    classI=training_data(training_data(:,65) == i,:);
    meanI = mean(classI(:,1:64));
    mean_data(i+1,:)= meanI;
    class_group_data{1,i+1}=i+1;
    class_group_data{2,i+1}=classI;
end
scatter_within=zeros(size(training_data,2)-1);
for x =1:10
    scatter_class=zeros(size(training_data,2)-1);
    for y = 1:size(class_group_data{2,x})
        class_row=transpose(class_group_data{2,x}(y,1:64));
        class_mean=transpose(mean_data(x,:));
        product = (class_row - class_mean) * transpose(class_row - class_mean);
        scatter_class=scatter_class+product;
    end
    scatter_within=scatter_within+scatter_class;
end
ovr_mean=mean(mean_data);
ovr_mean_t=transpose(ovr_mean);
scatter_between=zeros(size(training_data,2)-1);
for z=1:10
    class_size=size(class_group_data{2,z},1);
    class_mean_t=transpose(mean_data(z,:));
    difference_means=class_mean_t-ovr_mean_t;
    product_b = class_size * difference_means * transpose(difference_means);
    scatter_between = scatter_between+product_b;
end
scatter_within_inv = pinv(scatter_within);
result_matrix = scatter_within_inv*scatter_between;
[eigvector,eigvalue]=eig(result_matrix);
[eigvalues_sorted,indices]=sort(diag(eigvalue),'descend');
eigvector_sorted = eigvector(:,indices);
projection_matrix = eigvector_sorted;
end

