res = load('result');
U_rad = res(:,2);
a = 0.01372;
c = 29.98;
T_rad = sqrt(sqrt(U_rad/a/c));
plot(res(:,1), T_rad)
