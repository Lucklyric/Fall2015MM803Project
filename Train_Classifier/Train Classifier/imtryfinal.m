%Set these variables first.
num_images = 50;    %number of masks needed, make sure it is less than the total number of images.
index_first = 307;  %the index number of the image file produced using the openCV code.
ind = num2str(index_first);
fil = 'img_';
ext = '.jpg';   %change to required extension.

DISP = 0; %0 to suppress displaying images, 1 to display.

% outputs are a .mat file with name as specified in the last line of this
% code and Histsk, the histogram of the skin part.

for d = 1:1

fil = 'Image_13';    
filename = strcat(fil,ext);   
x = index_first;
xstr = num2str(x);
y = x+d;
ystr = num2str(y);
z = regexprep(filename,xstr,ystr);


imfile = imread(z);
imfileorig = imfile;

    %R,G,B components of the input image
    R = imfileorig(:,:,1);
    G = imfileorig(:,:,2);
    B = imfileorig(:,:,3);

    %Inverse of the Avg values of the R,G,B
    mR = 1/(mean(mean(R)));
    mG = 1/(mean(mean(G)));
    mB = 1/(mean(mean(B)));
    
    %Smallest Avg Value (MAX because we are dealing with the inverses)
    maxRGB = max(max(mR, mG), mB);
    
    %Calculate the scaling factors
    mR = mR/maxRGB;
    mG = mG/maxRGB;
    mB = mB/maxRGB;
   
    %Scale the values
     imfile(:,:,1) = R*mR;
     imfile(:,:,2) = G*mG;
     imfile(:,:,3) = B*mB;

Ci = rgb2ycbcr(imfile);
Hi = rgb2hsv(imfile);

mask = ones(size(imfile,1),size(imfile,2),1);

for i = 1:size(imfile,1)
    for j = 1:size(imfile,2)
        

       %R value thresholding
%       if (imfile(i,j,1) > 55)
%           mask(i,j,1) = 0;
%       end
        
         %if (imfile(i,j,1)/imfile(i,j,2) < 1.3 | imfile(i,j,1)/imfile(i,j,2) > 1.85)
             %imfile(i,j,1) = 0;
             %mask(i,j,1) = 0;
         %end
        
        %G value thresholding
         %if (imfile(i,j,2) < 40)
             %mask(i,j,1) = 0;
         %end
%         
%         %B value thresholding
         %if (imfile(i,j,3) < 20)
             %mask(i,j,1) = 0;
         %end
        
        % R-G value thresholding
%         if (imfile(i,j,1)-imfile(i,j,2) < 80 & imfile(i,j,1)-imfile(i,j,2) > 20)
%             mask(i,j,1) = 0;
%         end

        % CMYK thresholding
        if ((Ci(i,j,2) < 70 | Ci(i,j,2) > 130)) 
                        %R value thresholding
                        %if (imfile(i,j,1) > 55)
                            mask(i,j,1) = 0;
                        %end
        end
        
        if ((Ci(i,j,3) < 130) | (Ci(i,j,3) > 180))
                        %R value thresholding
                        %if (imfile(i,j,1) > 55)
                            mask(i,j,1) = 0;
                        %end
        end 
        
%         %HSV thresholding
%         % Value thresholding
%         if (Hi(i,j,3) > 0.4)
%             mask(i,j,1) = 1;
%         end
%         
%         %Saturation Thresholding
%         if (Hi(i,j,2) > 0.2 & Hi(i,j,2) < 0.6)
%             mask(i,j,1) = 1;
%         end
%         
%         %Hue thresholding
%         if (Hi(i,j,1) < 0.2 | Hi(i,j,1) > 0.8)
%             mask(i,j,1) = 1;
%         end
%         
        
        
    end
end

mask = im2uint8(mask);

Outimage = zeros(size(imfile,1),size(imfile,2),3);

for i = 1:size(imfile,1)
    for j = 1:size(imfile,2)
        if (mask(i,j,1) == 255)
            Outimage(i,j,1) = imfile(i,j,1);
            Outimage(i,j,2) = imfile(i,j,2);
            Outimage(i,j,3) = imfile(i,j,3);
        end
    end
end

Boutimage = im2bw(Outimage,0.5);

Boutimage = ~Boutimage;

if DISP
    figure(1),imshow(imfileorig)
    figure(2),imshow(imfile)
    figure(3),imshow(Boutimage)
    % figure(2), imshow(imfile2)
end

Histsk(:,:,d) = imhist(Boutimage);
% BW = edge(imfile(:,:,2),'prewitt',0.05);
% figure(3),imshow(BW)

Unclosed(:,:,d) = Boutimage;

SE = strel('octagon',6);    %rethink this scheme!
closed(:,:,d) = imclose(Boutimage,SE);
closed = ~closed;

conimage = closed(:,:,d);
conimageuint = uint8(conimage);

conimageuint(:,:,2) = conimageuint(:,:,1);
conimageuint(:,:,3) = conimageuint(:,:,1);

segimage(:,:,:,d) = conimageuint.*imfile;

concomp = bwconncomp(conimage);

% BW1 = bwperim(Boutimage);
%D = bwdist(Boutimage);
%OP(:,:,d) = bwareaopen(closed(:,:,d),2000);
% Dop = bwdist(OP);
statsfi = regionprops(concomp,'filledimage','filledarea');

for i = 1:size(statsfi,1)
    %if the area is less than 0.6 than it is possibly a hand!!!!!!
    area = ((statsfi(i).FilledArea)./(size(statsfi(i).FilledImage,1)*size(statsfi(i).FilledImage,2)));
    if area > 0.6
        %delete that part of the image
        conimage(concomp.PixelIdxList{i}) = 0;
    end
end
 
% numPixels = cellfun(@numel,concomp.PixelIdxList);
% [biggest,idx] = max(numPixels);
% conimage(concomp.PixelIdxList{idx}) = 0;

headless(:,:,:,d) = conimage;

conimageuint = uint8(conimage);

conimageuint(:,:,2) = conimageuint(:,:,1);
conimageuint(:,:,3) = conimageuint(:,:,1);

segheadlessimage(:,:,:,d) = conimageuint.*imfile;

if DISP
    figure(4),imshow(closed)
end

% save the mask image in a .mat file
% save maskdata.mat closed -append

% 3 heads, biggest components hence eliminate!
% CC = bwconncomp(Dop);
% numpixels = cellfun(@numel,CC.PixelIdxList);
% [biggest,idx] = max(numpixels);
% Dop(CC.PixelIdxList{idx}) = 0;
% 
% CC = bwconncomp(Dop);
% numpixels = cellfun(@numel,CC.PixelIdxList);
% [biggest,idx] = max(numpixels);
% Dop(CC.PixelIdxList{idx}) = 0;
% 
% CC = bwconncomp(Dop);
% numpixels = cellfun(@numel,CC.PixelIdxList);
% [biggest,idx] = max(numpixels);
% Dop(CC.PixelIdxList{idx}) = 0;
% 
% 
% figure(4),imshow(Dop)

fil = 'outimg_';    
filename = strcat(fil,xstr,ystr);   

imwrite(segheadlessimage(:,:,:,d),filename,'jpg');

fil = 'segimg_';    
filename = strcat(fil,xstr,ystr);   

imwrite(closed(:,:,d),filename,'jpg');

fil = 'headimg_';    
filename = strcat(fil,xstr,ystr);   

imwrite(segimage(:,:,:,d),filename,'jpg');

fil = 'headleimg_';    
filename = strcat(fil,xstr,ystr);   

imwrite(headless(:,:,:,d),filename,'jpg');

end


for i = 1:50
    imshow(segheadlessimage(:,:,:,i));
    pause(0.05)
end
% save jobsmaskfin.mat closed segheadlessimage segimage headless
