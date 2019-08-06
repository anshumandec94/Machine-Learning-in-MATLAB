function [princ_components,k]= myPCA(training_data)
    covariance_mat = cov(training_data);
    [vectors, diagmat] = eig(covariance_mat);
    [sorted_values,indices] = sort(diag(diagmat),'descend');
    sorted_vectors = vectors(:,indices);
    sum_of_variances = sum(sorted_values);
    [row,column]=size(training_data);
    proportion_of_var = [];
    for i=1:size(sorted_values,1)
        prop_of_var = sorted_values(i)/sum_of_variances;
        proportion_of_var(i)=prop_of_var;
    end
    pov_k =0;
    k =0;
    for j=1:size(proportion_of_var,2)
        pov_k=pov_k+proportion_of_var(j);
        if pov_k>0.9
            k=j;
            break;
        end
    end
    prev_sum = 0;
    sum_of_pov = [];
    for a=1:size(proportion_of_var,2)
        currpov = proportion_of_var(a);
        new_sum = prev_sum+currpov;
        sum_of_pov(a)=new_sum;
        prev_sum = new_sum;
    end
    xaxis=linspace(1,column,column);
    plot(xaxis,sum_of_pov);
    princ_components = sorted_vectors;
   k=j;
end

