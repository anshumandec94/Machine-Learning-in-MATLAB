function [p1,p2,pc1,pc2] = Bayesian_learning(train,valid)
class1=train(train(:,23)== 1,1:22);
class2=train(train(:,23)== 2,1:22);
p1=mean(class1,1);
p2=mean(class2,1);

possible_sigma = [-5 -4 -3 -2 -1 0 1 2 3 4 5];


valid_classes=valid(:,23);
valid_instances=valid(:,1:22);
min_error_rate=1;
min_sigma=1;
for i=1:size(possible_sigma,2)
    pclass1=1/(1+exp(-(possible_sigma(i))));
    pclass2=1-pclass1;
    
    error=0;
    
    for x=1:size(valid)
         product1=pclass1;
         product2=pclass2;
        
        for y=1:22
            product1=product1 * power(p1(y),valid_instances(x,y))*power((1-p1(y)),(1-valid_instances(x,y)));
            product2=product2 * power(p2(y),valid_instances(x,y))*power((1-p2(y)),(1-valid_instances(x,y)));
        end
        
       
        if(product1>=product2)
            class=1;
      
        else
            class=2;
        end
        if(valid_classes(x)~=class)
            
            error=error+1;
        end
        
        
    end
    error_rate=error/size(valid,1);
    sprintf('1. %d \t %f  \n',possible_sigma(i),error_rate*100)
    if(error_rate<min_error_rate)
        min_error_rate=error_rate;
        min_sigma=possible_sigma(i);
    end
end
sprintf('minimum sigma is %d',min_sigma)
pc1=1/(1+exp(-min_sigma));
pc2=1-pc1;




