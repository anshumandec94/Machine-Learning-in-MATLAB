training = load('face_train_data_960.txt');
testing = load('face_test_data_960.txt');
train_data = training(:,1:960);
train_classes = training(:,961);
test_data = testing(:,1:960);
test_classes = testing(:,961);
[principal_components,k] =myPCA(train_data);
Kvalues = [10 50 100];
figure()
first_five_faces = train_data(1:5,:);
meanarr = mean(train_data);
 idx=1;
for x=1:length(Kvalues)
    first_k_cmp = principal_components(:,1:Kvalues(x));
     
    projected_faces = first_five_faces*first_k_cmp;
    X = projected_faces * transpose(first_k_cmp);
    reconstruct_data = bsxfun(@plus, X, meanarr);
   
    for y = 1:size(reconstruct_data,1)
        subplot(4,5,idx);
        idx=idx+1;
        imagesc(reshape(reconstruct_data(y,1:end),32,30)');
        
    end
    
end