load('optdigits_train.txt')
load('optdigits_valid.txt')
load('optdigits_test.txt')
min_h=0;
min_val=1;
%loop for calling the mlp train on varying number of hidden units. 
%we also store the min_h, w and v values for the h value with least
%validation error. 
trainv=zeros(1,6);
validv=zeros(1,6);
hidden_units_candidates=[3 6 9 12 15 18];
count=0;
for i = [3 6 9 12 15 18]
    count=count+1;
    fprintf("For %d hidden units\n",i);
[z,w,v,train,val]=MLPtrain(optdigits_train,optdigits_valid,i,10);
    trainv(count)=train;
    validv(count)=val;
    if(min_val>val)
        min_val=val;
        min_h=i;
        min_w=zeros(i,65);
        min_w=w;
        min_v=zeros(10,i+1);
        min_v=v;
    end
end
figure
hold on
plot(hidden_units_candidates,trainv);
plot(hidden_units_candidates,validv);
%we call the mlptest function on the test data for our minimum h calculated
%before. 
fprintf("Error rate on test set for %d \n",min_h);
[~,err]= MLPtest(optdigits_test,min_w,min_v);
fprintf("Error rate is %d \n",err);