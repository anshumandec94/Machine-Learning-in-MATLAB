[mean_1,mean_2,cov_mat1,cov_mat2,total_err] = Q3a()
disp('model 1')
disp('m1')
mean_1
disp('m2')
mean_2
disp('s1')
cov_mat1
disp('s2')
cov_mat2
disp('error')
total_err
[mean_1,mean_2,cov_mat1,cov_mat2,total_err] = Q3b()
disp('model 2')
disp('m1')
mean_1
disp('m2')
mean_2
disp('s1')
cov_mat1
disp('s2')
cov_mat2
disp('error')
total_err
[mean_1,mean_2,diag1,diag2,total_err] = Q3c()
disp('model 3')
disp('m1')
mean_1
disp('m2')
mean_2
disp('s1')
diag1
disp('s2')
diag2
disp('error')
total_err

[mean_1,mean_2,eig1,eig2,total_err] = Q3d()

disp('model 4')
disp('m1')
mean_1
disp('m2')
mean_2
disp('alpha1')
eig1
disp('alpha2')
eig2
disp('error')
total_err

function [mean_1,mean_2,cov_mat1,cov_mat2,total_err] = Q3a()
load('training_data.txt');
class1 = training_data( training_data(:,9) == 1, 1:8);
class2 = training_data( training_data(:,9) == 2, 1:8);
prior1 = 0.6;
prior2 = 0.4;
mean_1 = mean(class1);
mean_2 = mean(class2);
pred_class = 0;
cov_mat1 = cov(class1);
cov_mat2 = cov(class2);
error = 0;
load('test_data.txt');
[m,n] = size(test_data);
class_sample = zeros(m);
denom1 = 1/((2*pi)^(4)*sqrt(det(cov_mat1)));
denom2 = 1/((2*pi)^(4)*sqrt(det(cov_mat2)));
for i = 1:m
    sample = test_data(i,1:8);
    term1 = exp((-0.5*(sample-mean_1)*inv(cov_mat1)*(sample-mean_1)'));
    term2 = exp((-0.5*(sample-mean_2)*inv(cov_mat2)*(sample-mean_2)'));
    g1 = (term1*denom1)*prior1;
    g2 = (term2*denom2)*prior2;
    class_sample(i) = log(g1/g2);
    if(class_sample(i) > 0)
        pred_class = 1;
        
    end
    if(class_sample(i) < 0)
        pred_class = 2;
    end
    if (pred_class ~= test_data(i,9))
        error = error+1;
    end
    
end
total_err = error/m;

end

function [mean_1,mean_2,cov_mat1,cov_mat2,total_err] = Q3b()
load('training_data.txt');
class1 = training_data( training_data(:,9) == 1, 1:8);
class2 = training_data( training_data(:,9) == 2, 1:8);
prior1 = 0.6;
prior2 = 0.4;
mean_1 = mean(class1);
mean_2 = mean(class2);
pred_class = 0;
cov_mat1 = cov(class1);
cov_mat2 = cov(class2);
cov_mat = (cov_mat1*prior1) + (cov_mat2*prior2);
error = 0;
load('test_data.txt');
[m,n] = size(test_data);
class_sample = zeros(m);
denom1 = 1/((2*pi)^(4)*sqrt(det(cov_mat)));
denom2 = 1/((2*pi)^(4)*sqrt(det(cov_mat)));
for i = 1:m
    sample = test_data(i,1:8);
    term1 = exp((-0.5*(sample-mean_1)*inv(cov_mat)*(sample-mean_1)'));
    term2 = exp((-0.5*(sample-mean_2)*inv(cov_mat)*(sample-mean_2)'));
    g1 = (term1*denom1)*prior1;
    g2 = (term2*denom2)*prior2;
    class_sample(i) = log(g1/g2);
    if(class_sample(i) > 0)
        pred_class = 1;
        
    end
    if(class_sample(i) < 0)
        pred_class = 2;
    end
    if (pred_class ~= test_data(i,9))
        error = error+1;
    end
    
end
total_err = error/m;
end
function [mean_1,mean_2,diag1,diag2,total_err] = Q3c()
load('training_data.txt');
class1 = training_data( training_data(:,9) == 1, 1:8);
class2 = training_data( training_data(:,9) == 2, 1:8);
prior1 = 0.6;
prior2 = 0.4;
mean_1 = mean(class1);
mean_2 = mean(class2);
pred_class = 0;
cov_mat1 = cov(class1);
cov_mat2 = cov(class2);
diag1 = diag(diag(cov_mat1));
diag2 = diag(diag(cov_mat2));

error = 0;
load('test_data.txt');
[m,n] = size(test_data);
class_sample = [];
denom1 = 1/((2*pi)^(4)*sqrt(det(diag1)));
denom2 = 1/((2*pi)^(4)*sqrt(det(diag2)));
for i = 1:m
    sample = test_data(i,1:8);
    term1 = exp((-0.5*(sample-mean_1)*inv(diag1)*(sample-mean_1)'));
    term2 = exp((-0.5*(sample-mean_2)*inv(diag2)*(sample-mean_2)'));
    g1 = (term1*denom1)*prior1;
    g2 = (term2*denom2)*prior2;
    class_sample(i) = log(g1/g2);
    if(class_sample(i) > 0)
        pred_class = 1;
        
    end
    if(class_sample(i) < 0)
        pred_class = 2;
    end
    if (pred_class ~= test_data(i,9))
        error = error+1;
    end
    
end
total_err = error/m;
end
function [mean_1,mean_2,eig1,eig2,total_err] = Q3d()
load('training_data.txt');
class1 = training_data( training_data(:,9) == 1, 1:8);
class2 = training_data( training_data(:,9) == 2, 1:8);
prior1 = 0.6;
prior2 = 0.4;
mean_1 = mean(class1);
mean_2 = mean(class2);
pred_class = 0;
cov_mat1 = cov(class1);
cov_mat2 = cov(class2);
eig1 = eig(cov_mat1);
eig2 = eig(cov_mat2);

dimat1 =diag(eig1);
dimat2 = diag(eig2);


error = 0;
load('test_data.txt');
[m,n] = size(test_data);
class_sample = [];
denom1 = 1/((2*pi)^(4)*sqrt(det(dimat1)));
denom2 = 1/((2*pi)^(4)*sqrt(det(dimat2)));
for i = 1:m
    sample = test_data(i,1:8);
    term1 = exp((-0.5*(sample-mean_1)*inv(dimat1)*(sample-mean_1)'));
    term2 = exp((-0.5*(sample-mean_2)*inv(dimat2)*(sample-mean_2)'));
    g1 = (term1*denom1)*prior1;
    g2 = (term2*denom2)*prior2;
    class_sample(i) = log(g1/g2);
    if(class_sample(i) > 0)
        pred_class = 1;
        
    end
    if(class_sample(i) < 0)
        pred_class = 2;
    end
    if (pred_class ~= test_data(i,9))
        error = error+1;
    end
    
end
total_err = error/m;
end

