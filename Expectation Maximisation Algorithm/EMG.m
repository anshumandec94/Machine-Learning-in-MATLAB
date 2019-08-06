function [responsibility,mus,lnsumM] = EMG(impath,flag,k)
[img , cmap]=imread(impath);
img_rgb=ind2rgb(img,cmap);
img2double=im2double(img_rgb);
dataset = reshape(img2double,[],3);
N=size(dataset,1);
[idx,mus] = kmeans(dataset,k,'MaxIter',3);
sz= size(dataset);
pi_vals = zeros(1,k);
pi_vals(:)=1/k;
sigmak=zeros(sz(2),sz(2),k);
responsibility = zeros(sz(1),k);
lambda = 0.1;
for i=1:k
      sigmak(:,:,i)=cov(dataset(idx==i,:));
      
end
lnsumE=zeros(100,1);
figure
lnsumM=zeros(100,1);

converged=0;
itr=1;
while (itr<=100 && converged == 0)

 numerator = zeros(N,k);
denominator = zeros(N,1);   
for a=1:N
    for b=1:k 
       
    numerator(a,b) = pi_vals(:,b)*mvnpdf(dataset(a,:),mus(b,:),sigmak(:,:,b));
    denominator(a) = denominator(a) + numerator(a,b);
    end
end
%Expectation step
numerator(numerator==0)=eps;
for a=1:N
    for b=1:k
        responsibility(a,b)=numerator(a,b)/denominator(a);
    end
    
end
Nk=zeros(1,k);
for i=1:N
    
    for j=1:k
        Nk(1,j)=Nk(1,j)+responsibility(i,j);
    end
end
lnsum=0;
%calculating log likelihood Exepectation
for a=1:N
    for b=1:k
        lnsum=lnsum+(pi_vals(:,b)*mvnpdf(dataset(a,:),mus(b,:),sigmak(:,:,b)));
    end
        lnsumE(itr)=lnsumE(itr)+log(lnsum);
end

%Maximization

pi_vals = Nk/N;

for j=1:k
    sum=zeros(1,3);
    sums=zeros(3,3);
    for i=1:N
        sum=sum+(responsibility(i,j)*dataset(i,:));
    end
    mus(j,:)=sum/Nk(1,j);
    for i=1:N
        sample=dataset(i,:);
    
        sums=sums+(responsibility(i,j)*((sample-mus(j,:))'*(sample-mus(j,:))));
    
    end

    sigmak(:,:,j)=sums/Nk(1,j);
    if(flag == 1)
        sigmak(:,:,j)=sigmak(:,:,j)+((lambda/Nk(1,j))*eye(3));
    end
end
%calculating log likelihood maximization

lnsum=0;
for a=1:N
    for b=1:k
        lnsum=lnsum+(pi_vals(:,b)*mvnpdf(dataset(a,:),mus(b,:),sigmak(:,:,b)));
    end
    lnsumM(itr)=lnsumM(itr)+log(lnsum);
end
    if itr>1
        diff = abs(lnsumM(itr)-lnsumM(itr-1));
        if (diff <= 0.1)
            converged = 1;
        end
            
    end

itr=itr+1;
end

hold on
%plotting log likelihood vs iterations 
plot(1:itr-1,lnsumE(1:itr-1),'o');
plot(1:itr-1,lnsumM(1:itr-1),'*');
hold off
%plotting compressed image
cluster_indices = zeros(N,1);
for i=1:N
    [values, indices]=max(responsibility(i,:));
    cluster_indices(i)=indices;
end

colour_values = zeros(N,3);
for d=1:N
    colour_values(d,:) = mus(cluster_indices(d),:);
end
[row,column,depth]=size(img2double);
figure
cmp_image = reshape(colour_values,row,column,depth);
imshow(cmp_image);
sprintf("For k =%d ",k);
sprintf("The h matrix")
disp(responsibility)
sprintf("Cluster means")
disp(mus)
sprintf("Log likelihood values")
disp(lnsumM(1:itr-1))


end


