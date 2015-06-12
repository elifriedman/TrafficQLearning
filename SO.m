cd('/home/friedm3/Dropbox/TrafficNetworks/CodeBase')
costmat = load('cost_mat_mod.csv');
costs = load('costs_modified.csv');

AL = zeros(1,32);
AL(1:8) = 1;
AM = circshift(AL',8)';
BL = circshift(AM',8)';
BM = circshift(BL',8)';

cvx_begin
    variable X(32,1) integer nonnegative
    minimize transpose(costmat*X + costs)*X;
    AL*X == 600;
    AM*X == 400;
    BL*X == 300;
    BM*X == 400;
cvx_end