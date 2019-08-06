function [] = Bayesian_testing(test,p1,p2,pc1,pc2)
    test_instances=test(:,1:22);
    test_classes=test(:,23);
    error=0;
    for x=1:length(test_instances)
        product1=pc1;
        product2=pc2;
        for y=1:22
            product1=product1 * power(p1(y),test_instances(x,y))*power((1-p1(y)),(1-test_instances(x,y)));
            product2=product2 * power(p2(y),test_instances(x,y))*power((1-p2(y)),(1-test_instances(x,y)));
        end
         if(product1>=product2)
            class=1;
      
        else
            class=2;
         end
        if(test_classes(x)~=class)
            
            error=error+1;
        end
        
    end
    error_rate=error/length(test_instances);
    sprintf('The error rate for the test data with best sigma is %f percent ',error_rate*100)
    
end

