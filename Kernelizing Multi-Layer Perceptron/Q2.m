rng(1); % For reproducibility
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1); % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
r2 = sqrt(3*rand(100,1)+2); % Radius
t2 = 2*pi*rand(100,1); % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

figure; 
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on 
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15) 
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

%calculating the gram matrix
data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;
q=3;
[alphaval,betaval]=kernPercGD(data3,theclass,q);
N=size(data3,1);
d = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)]; 
scores=zeros(size(xGrid,1),1);
for i=1:size(xGrid,1)
    sample=xGrid(i,:);
    sum_i=0;
    for j=1:N
        inner_product=data3(j,:)*sample';
        inner_product=inner_product+1;
        inner_product=inner_product^q;
        sum_i=sum_i+(alphaval(j)*theclass(j)*inner_product);
    end
    sum_i=sum_i+betaval;
    scores(i)=sum_i;
end

figure;
 gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0 0],'k');

cl2 = fitcsvm(data3,theclass,'KernelFunction','polynomial','PolynomialOrder',q,'BoxConstraint',1);
[~,scores2]=predict(cl2,xGrid);
contour(x1Grid,x2Grid,reshape(scores2(:,2),size(x1Grid)),[0 0],'r');
legend({'-1','+1','originalclassboundary','perc','svm'});
load('optdigits79_train.txt');
train=optdigits79_train(:,1:end-1);
class79=optdigits79_train(:,end);

[alpha2,beta2]=kernPercGD(train,class79,q);
train_err_rate=0;
%classifying train data
for i=1:size(train,1)
    sum_i=0;
    
    sample=train(i,:);
    for j=1:size(train,1)
        inner_product=sample*train(j,:)';
        inner_product=inner_product+1;
        inner_product=inner_product^q;
        sum_i=sum_i+(alpha2(j)*class79(j)*inner_product);
    end
    sum_i=sum_i+beta2;
    predicted=sign(sum_i);
    if predicted~=class79(i)
        train_err_rate=train_err_rate+1;
    end
        
end
train_err_rate=train_err_rate/size(train,1);
fprintf("train error rate for 79 data is %f \n",train_err_rate*100);
test_err_rate=0;
load('optdigits79_test.txt');
test=optdigits79_test(:,1:end-1);
testclass=optdigits79_test(:,end);

for i=1:size(test,1)
    sum_i=0;
    
    sample=test(i,:);
    for j=1:size(train,1)
        inner_product=sample*train(j,:)';
        inner_product=inner_product+1;
        inner_product=inner_product^q;
        sum_i=sum_i+(alpha2(j)*class79(j)*inner_product);
    end
    predicted=sign(sum_i);
    if predicted~=testclass(i)
        test_err_rate=test_err_rate+1;
    end
        
end
test_err_rate=test_err_rate/size(test,1);
fprintf("The test error rate for 79 is %f\n",test_err_rate*100);

load('optdigits49_train.txt');
train=optdigits49_train(:,1:end-1);
class49=optdigits49_train(:,end);
load('optdigits49_test.txt');
test=optdigits49_test(:,1:end-1);
testclass=optdigits49_test(:,end);

[alpha3,beta3]=kernPercGD(train,class49,q);
train_err_rate=0;
for i=1:size(train,1)
    sum_i=0;
    
    sample=train(i,:);
    for j=1:size(train,1)
        inner_product=sample*train(j,:)';
        inner_product=inner_product+1;
        inner_product=inner_product^q;
        sum_i=sum_i+(alpha3(j)*class49(j)*inner_product);
    end
    sum_i=sum_i+beta3;
    predicted=sign(sum_i);
    if predicted~=class79(i)
        train_err_rate=train_err_rate+1;
    end
        
end
train_err_rate=train_err_rate/size(train,1);
fprintf("The error rate for 49 data training is %f\n",train_err_rate*100);
test_err_rate=0;
for i=1:size(test,1)
    sum_i=0;
    
    sample=test(i,:);
    for j=1:size(train,1)
        inner_product=sample*train(j,:)';
        inner_product=inner_product+1;
        inner_product=inner_product^q;
        sum_i=sum_i+(alpha3(j)*class49(j)*inner_product);
    end
    sum_i=sum_i+beta3;
    predicted=sign(sum_i);
    if predicted~=testclass(i)
        test_err_rate=test_err_rate+1;
    end
        
end
test_err_rate=test_err_rate/size(test,1);
fprintf("The error rate for 49 data testing is %f\n",test_err_rate*100);

