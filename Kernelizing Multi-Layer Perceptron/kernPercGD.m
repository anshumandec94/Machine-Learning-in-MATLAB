function [alphaval,betaval] = kernPercGD(train,train_class,degree)
N=size(train,1);

gramMat=zeros(N);
for i=1:N
    for j=1:N
        gramMat(i,j)=train(i,:)*train(j,:)';
        gramMat(i,j)=gramMat(i,j)+1;
        gramMat(i,j)=gramMat(i,j)^degree;
        
    end
        
end
alphaval=zeros(N,1);
y=zeros(N,1);
betaval=0;
error=0;
prev_error=0;
err=zeros(200,1);

converged=0;
k=0;
while (converged==0)
    k=k+1;
    for i=1:N
        sum=0;
        for j=1:N
            sum=sum+(alphaval(j,1)*train_class(j)*gramMat(i,j));
        end
        sum=sum+betaval;
        if(sum*train_class(i)<=0)
            alphaval(i)=alphaval(i)+1;
            betaval=betaval+train_class(i);
            error=error+(1-(sum*train_class(i)));
        end
        y(i)=sum*train_class(i);
    end
    if(error==0)
        converged=1;
    end
    if(abs(prev_error-error)==0)
        converged=1;
    end
    prev_error=error;
    err(k)=error;
    error=0;
    
    
    
    
end
end

