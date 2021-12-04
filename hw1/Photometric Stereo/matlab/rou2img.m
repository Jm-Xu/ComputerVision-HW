% Re-organize from vector to image 

function N_est_img = rou2img(N_est, height, width, m)

p = length(m);

n_x = zeros(height*width, 1);
for i = 1 : p
    n_x(m(i)) = N_est(i, 1);
end

n_x = reshape(n_x, height, width);

N = zeros(height, width, 1);
N(:, :, 1) = n_x;
N(isnan(N)) = 0;

N_est_img = N;



