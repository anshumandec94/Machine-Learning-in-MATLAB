function [z,w,v,train_err,valid_err] = MLPtrain(train,val,m,k)
    d=size(train,2);
    learning_rate=10^-5;
    %Initializing w and v[m,d][k,m]
    lwr=-0.01;
    upr=0.01;
    w=((upr-lwr).*rand(m,d))+lwr;
    v=((upr-lwr).*rand(k,m))+lwr;
    z=zeros(size(train,1),m);
    %adding the d+1th column to input
    i_bias= ones(size(train,1),1);
    h_bias=ones(k,1);
    v=[v h_bias];
    resp = train(:,65);
    train = train(:,1:64);
    train =[train i_bias resp];
    [r,c] =size(train);
    error=0;
    prev_error=0;
    converged=false;
    round=0;
    while ((converged == false) || (round<1000))
        shuffle=randperm(r);
        train=train(shuffle,:);
        [r,c]=size(train);
        if (isnan(error)==true)
            fprintf("round %d",round)
            break
        end
        round=round+1;
        if(round>800)
            learning_rate=10^-8;
        end
        
        
       for itr=1:r
       
        
             
            y=zeros(1,k);
            res=zeros(1,k);
            hidden_vals=[z(itr,:) 1];
            sample=train(itr,1:c-1);
             output=train(itr,c);
            resp=zeros(1,k);
            resp(output+1)=1;
            update_v=zeros(k,m+1);
            update_w=zeros(m,d);
            %calculating hidden unit values or zih
            for h=1:m
                ss=0;
                for j=1:size(sample,2)
                    ss=ss+(sample(j)*w(h,j));
                end
                hidden_vals(h)=max([ss 0]);
            end
            %calculating our oi values for the sample
            for ix=1:k
                res(1,ix)=hidden_vals*v(ix,:)';
            end
            %using softmax to get class predictions
            for ix=1:k
                y(1,ix)=exp(res(1,ix))/sum(exp(res));
            end
            for ix=1:k
                %calculating our update value for v    
                update_v(ix,:)=(learning_rate*((resp(1,ix)-y(1,ix))*hidden_vals));
                
            end
            for h=1:m
                for j=1:c-1
                    sum_resp=0;
                    for ix=1:k
                        sum_resp=sum_resp+(((resp(ix)-y(ix))*v(ix,h)));
                    end
                    %calculating w update values
                    update_w(h,j)=(learning_rate*sum_resp*sample(j));
                end
            end
            %performing our update
            v=v+update_v;
            w=w+update_w;
            z(itr,:)=hidden_vals(:,1:end-1);
            %calculating our cost function
            for ix=1:k
                if y(ix)==0
                    continue
                end
                error=error+(resp(ix)*log(y(ix)));
            end
            
           
                
           
       end
            %convergence check
            if abs(prev_error-error)<0.001
              
               converged = true;
               break;
            end
            prev_error=error;
            error=0;
       
        
       
        
    end
    fprintf("converged error %d\n",prev_error);
    %calling the mlp function to calculate prediction error for train and
    %validate which are printed and returned
  [~,train_err_rate]= MLPtest(train,w,v);
    [~,val_err_rate]= MLPtest(val,w,v);
    fprintf("Train error_rate %d\n",train_err_rate*100);
    fprintf("Validation error rate %d\n",val_err_rate*100);
    train_err=train_err_rate;
    valid_err=val_err_rate;
    

end

