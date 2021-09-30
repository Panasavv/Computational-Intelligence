clear variables;
clc;

DCMotor1 = readfis('DC-Motor-1');
DCMotor1_Simulation = DCMotor1;
fuzzyLogicDesigner(DCMotor1);
gensurf(DCMotor1);             
view([-120 30]);

writefis(DCMotor1,'DC-Motor-1');
