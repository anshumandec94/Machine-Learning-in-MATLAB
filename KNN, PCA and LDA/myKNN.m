function [error_rate] = myKNN(training_features,test_instances,training_classes,test_classes,k)
    error = 0;
    for x=1:size(test_classes,1)
        neighbors = zeros(size(training_features,2),2);
        instance = test_instances(x,:);
        class = test_classes(x);
        for y=1:size(training_features,1)
            distance = norm(instance - training_features(y,:));
            neighbors(y,1)=distance;
            neighbors(y,2)=training_classes(y);
            
        end
        neighbors_sorted = sortrows(neighbors,1,'ascend');
        neighbors_sorted_top_k = neighbors_sorted(1:k,2);
        pred = mode(neighbors_sorted_top_k);
        if ~(pred == class)
            error = error + 1;
        end
        
        
    end
        error_rate = (error/size(test_instances,1))*100;
    
end

