function [I,S] = mySelect(pixs, data ,idx, low_p, high_p)
% Dealing  with  shadows  and  highlights
[sort_pixs, index] = sort(pixs(:,idx));
sel_b = max(1, floor(size(data.imgs,1) * low_p));
sel_e = min(size(data.imgs,1), ceil(size(data.imgs,1) * high_p)); 
sel_idx = index(sel_b: sel_e);
I = sort_pixs(sel_b: sel_e,:);
S = data.s(sel_idx,:);
end

