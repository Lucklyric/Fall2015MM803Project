

clear all;
close all;
clc;
p=imread('handdetection.jpg');
[h,w] = size(p);
q=rgb2ycbcr(p);
a=q(:,:,2);
b=q(:,:,3);


for i=1:470
    for j=1:600
        if((a(i,j)>=0)&&(b(i,j)>=120)&&(a(i,j)<=0)&&(b(i,j)<=200))
            img(i,j)=1;
        else img(i,j)=0;
        end
    end
end
imshow(img);

