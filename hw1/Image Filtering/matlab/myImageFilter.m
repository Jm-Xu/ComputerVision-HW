function [img1] = myImageFilter(img0, h)
img1 = zeros('like',img0);
pad_r = floor(size(h,1)/2);
pad_c = floor(size(h,2)/2);
pad_img = padarray(img0, [pad_r, pad_c], 'replicate');
for i = 1: size(img0,1)
    for j = 1: size(img0,2)
        centi = i + pad_r;
        centj = j + pad_c;
        img1(i,j) = sum(sum(h .* pad_img(centi-pad_r: centi+pad_r, centj-pad_c: centj+pad_c)));
    end
end
