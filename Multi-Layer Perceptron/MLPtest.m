function [z,error_rate] = MLPtest(test,w,v)
    [r,c]=size(test);
    [k,m]=size(v);
    z=zeros(r,m-1);
    
    error=0;
    for itr=1:r
        y=zeros(1,k);
        ind=0;
        res=zeros(1,k);
        sample=test(itr,1:c-1);
        response=test(itr,c);
        %for input bias
        hidden_vals =[z(itr,:) 1];
        %calculating our z-values
        for h=1:m-1
                ss=0;
                for j=1:size(sample,2)
                    ss=ss+(sample(j)*w(h,j));
                end
                hidden_vals(h)=max([ss 0]);
               
        end
         %calculating o
         for ix=1:k
                res(1,ix)=hidden_vals*v(ix,:)';
         end
         %calculating probabilities
         for ix=1:k
                y(1,ix)=exp(res(1,ix))/sum(exp(res));
         end
         %our prediction is the highest probabilty of the class for the
         %sample
         [~,ind]=max(y);
         predicted=ind-1;
         
         if(predicted~=response)
             error=error+1;
         end
         z(itr,:)=hidden_vals(1:end-1);
         
            
            
        
    end
    error_rate=error/r;
    
    z=z(:,1:end-1);
    
end

