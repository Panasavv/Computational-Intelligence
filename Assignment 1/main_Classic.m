clear variables;
clc;

load("PIController.mat");
Gp = zpk([], [-1 -9], 10);

sysOpenLoop = PIController*Gp;
sys = feedback(PIController*Gp,1);
step(sys);

save("PIController.mat",'PIController');
