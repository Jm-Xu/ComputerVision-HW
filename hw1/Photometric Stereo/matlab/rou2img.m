% Re-organize from vector to image 

function rou_img = rou2img(rou, height, width, m)

p = length(m);
n_x = zeros(height*width, 1);
rou = rescale(rou);

for i = 1 : p
    n_x(m(i)) = rou(i, 1);
end

n_x = reshape(n_x, height, width);

N = zeros(height, width, 1);
N(:, :, 1) = n_x;
N(isnan(N)) = 0;

rou_img = N;



