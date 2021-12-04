function [img1] = NMS(amp,dirx)
%NMS 此处显示有关此函数的摘要
%   此处显示详细说明
img1 = zeros('like', amp);
for i = 1: size(amp,1)
    for j = 1: size(amp,2)
    if dirx < -3 * pi / 8 || dirx > 3 * pi / 8 
        if i - 1 >= 1 && amp(i,j) < amp(i-1,j)
            img1(i,j) = 0;
        elseif i + 1 <= size(amp,1) && amp(i,j) < amp(i+1,j)
            img1(i,j) = 0;
        else
            img1(i,j) = amp(i,j);
        end
    elseif dirx < -pi / 8
        if i - 1 >= 1 && j - 1 >= 1 && amp(i,j) < amp(i-1,j-1)
            img1(i,j) = 0;
        elseif i + 1 <= size(amp,1) && j + 1 <= size(amp,2) && amp(i,j) < amp(i+1,j+1)
            img1(i,j) = 0;
        else
            img1(i,j) = amp(i,j);
        end
    elseif dirx < pi / 8
        if j - 1 >= 1 && amp(i,j) < amp(i,j-1)
            img1(i,j) = 0;
        elseif j + 1 <= size(amp,2) && amp(i,j) < amp(i,j+1)
            img1(i,j) = 0;
        else
            img1(i,j) = amp(i,j);
        end
    else 
        if i - 1 >= 1 && j + 1 <= size(amp,2) && amp(i,j) < amp(i-1,j+1)
            img1(i,j) = 0;
        elseif i + 1 <= size(amp,1) && j - 1 >= 1 && amp(i,j) < amp(i+1,j-1)
            img1(i,j) = 0;
        else
            img1(i,j) = amp(i,j);
        end
    end    
    end
        
end

