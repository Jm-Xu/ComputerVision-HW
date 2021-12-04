function [N, R, normal, rou] = L2_PMS(data, m)
pixs = zeros(size(data.imgs,1), length(m), 3);
for i = 1:size(data.imgs,1)
    pixs(i,:,:) = normal_img2vec(cell2mat(data.imgs(i)), m);
end
normal = zeros(length(m), 3);
rou = zeros(length(m), 3);
% imgmat = cell2mat(data.imgs(1:3));
% data.imgs(1:3) * ((data.s(1:3) .* data.L(1:3, 1))^(-1));

for color = 1:3
for idx = 1:length(m)
    [I, S, L] = mySelect(pixs, data ,idx, color, 1/5, 4/5);
    % I = reshape(pixs(1:3,idx,1),[],1);
    % rou_n = ((data.s(1:3) .* data.L(1:3, 1))^(-1)) * I;
    sl = S .* L;
    rou_n = (sl.' * sl)^-1 * sl.' * I;
    normal(idx,:) = rou_n ./ norm(rou_n);
    rou(idx, color) = norm(rou_n);
end
end
N = normal_vec2img(normal, size(data.mask,1), size(data.mask,2), m);
R = rou2img(rou, size(data.mask,1), size(data.mask,2), m);
end

