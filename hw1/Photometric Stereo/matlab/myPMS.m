function [N, R, normal, rou] = L2_PMS(data, m)
pixs = zeros(size(data.imgs,1), length(m));
for i = 1:size(data.imgs,1)
    E = imread_datadir_re(data,i);
    vec =  normal_img2vec(E, m);
    Inten = 0.3*vec(:,1) + 0.6*vec(:,2) + 0.1*vec(:,3);
    pixs(i,:) = Inten;
end
normal = zeros(length(m), 3);
rou = zeros(length(m), 1);

for idx = 1:length(m)
    % Dealing  with  shadows  and  highlights
    [I, S] = mySelect(pixs, data ,idx, 1/5, 4/5);
    rou_n = (S.' * S)^-1 * S.' * I;
    normal(idx,:) = rou_n ./ norm(rou_n);
    rou(idx,1) = norm(rou_n);
end

N = normal_vec2img(normal, size(data.mask,1), size(data.mask,2), m);
R = rou2img(rou, size(data.mask,1), size(data.mask,2), m);
end

