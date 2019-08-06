%Calling function on stadium.bmp for the given k values
k=[4,8,12];
for i=1:length(k)
    [h,m,l]=EMG('stadium.bmp',0,k(i));
end
%calling EMG for k=7 on goldy.bmp
[resp,mus,likelihood]=EMG('goldy.bmp',0,7);
%calling Kmeans for k=7 on goldy.bmp
[img , cmap]=imread('goldy.bmp');
img_rgb=ind2rgb(img,cmap);
img2double=im2double(img_rgb);
dataset = reshape(img2double,[],3);
[idx,mus]= kmeans(dataset,7);
N=length(dataset);
colour_values = zeros(N,3);
for d=1:N
    colour_values(d,:) = mus(idx(d),:);
end
[row,column,depth]=size(img2double);
figure
cmp_image = reshape(colour_values,row,column,depth);
imshow(cmp_image);
%calling EM on goldy with k=7 and regularisation activated. 
[resp,mus,likelihood]=EMG('goldy.bmp',1,7);