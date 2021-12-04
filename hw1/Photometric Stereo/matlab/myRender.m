function [pixs] = myRender(data, m, idx, normal, rou)
% re-render picture of the object with recovered normal and albedo
ori_pixs = normal * data.s(idx, :).' .* rou .* data.L(idx,:);
ori_pixs = max(ori_pixs, 0);
pixs = normal_vec2img(ori_pixs, size(data.mask,1), size(data.mask,2), m);
end

