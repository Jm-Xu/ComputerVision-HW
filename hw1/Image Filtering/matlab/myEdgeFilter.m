function [img1] = myEdgeFilter(img0, sigma)
%Your implemention
g_ker = fspecial('gaussian', 2 * ceil(3 * sigma) + 1, sigma);
smo_img = myImageFilter(img0, g_ker);
gradx = myImageFilter(smo_img, fspecial('sobel').');
grady = myImageFilter(smo_img, fspecial('sobel'));
amp = (gradx.^2 + grady.^2).^(1/2);
dirx = atan(1);
img1 = NMS(amp, dirx);
end
    
                
        
        
