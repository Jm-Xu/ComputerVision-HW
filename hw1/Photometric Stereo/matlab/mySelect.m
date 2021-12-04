function [I,S,L] = mySelect(pixs, data ,idx, c, low_p, high_p)
%MY 此处显示有关此函数的摘要
%   此处显示详细说明
[sort_pixs, index] = sort(pixs(:,idx,c));
sel_b = max(1, floor(size(data.imgs,1) * low_p));
sel_e = ceil(size(data.imgs,1) * high_p); 
sel_idx = index(sel_b: sel_e);
I = sort_pixs(sel_b: sel_e,:);
S = data.s(sel_idx,:);
L = data.L(sel_idx,c);

% sort_pixs = pixs(:,idx,3);
% I = sort_pixs(1:3,:);
% S = data.s(1:3,:);
% L = data.L(1:3,3);
end

