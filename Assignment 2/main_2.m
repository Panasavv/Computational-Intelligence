clear variables;
clc;

%Initiallize everything
x_start = 3.8;
y_start = 0.5;
%Final Destination
destination = [10 3.2];
%Speed
speed = 0.05;
%Degrees
angles_init = [0 45 -45];
iter=2;

%Checker checks if we have to break out of loop
checker = 0;
%Our diagram
Vehicle=figure;

Car_C = readfis('Car_C_final.fis');

% Main



for angles_index= 1:3
    angles = angles_init(angles_index);
    x =[x_start];
    y =[y_start];
    
    while ~checker
        
        dh=10;
        dv=5;

        %This is the distance from wall
        x_check=x(iter-1);
        y_check=y(iter-1);
        if x_check < 0 || y_check < 0 || x_check > 10 || y_check > 5 % Outside of our lines
            dh = -1;
            dv = -1;
        elseif x_check < 5 %before the first line of our obstacle
            dv = y_check;
            if y_check <= 1
                dh = (5 - x_check);
            elseif y_check <= 2
                dh = (6 - x_check);
            elseif y_check <= 3
                dh = (7 - x_check);
            end
        elseif x_check < 6 && ~(y_check <= 1) %between first and second line of obstacle
            if y_check <= 2
                dh = (6 - x_check);
            elseif y_check <= 3
                dh = (7 - x_check);
            end
            dv = (y_check - 1);
        elseif x_check < 7 && ~(y_check <= 2) %between second and third line of obstacle
            if y_check <= 3
                dh = (7 - x_check);
            end
            dv = (y_check - 2);
        elseif x_check >= 7 && ~(y_check <= 3) %above third line of obstacle
            dv = (y_check - 3);
        else
            dh = -1;
            dv = -1;
        end
        
        %If we are out of bounds we break
        if dh < 0 || dh > 10 || dv < 0 || dv > 5
            checker= 1;
            break;
        end
        
        %Evaluattion point
        eval_angles = evalfis([dh/10 dv/5 angles], Car_C); 
    
        %Angles
        k = angles + eval_angles;
        if k < -180
            k = 360 - k;
        elseif k > 180
            angles = k - 360;
        else
            angles = k;
        end
        
        %New Position
        x = [x x(iter-1) + speed*cosd(angles)];
        y = [y y(iter-1) + speed*sind(angles)];
    
        %Check if we are done and if so, break
        if x(iter) == destination(1)
            if y(iter) == destination(2)
                checker = 1;
            end
        end

        %if iter == 30000
        %    checker = 1;
        %end
    
        iter = iter+1;
    end
    
    %Plotting
    subplot(2,2,angles_index);
    
    %Obstacles
    plot(linspace(0,1) * 0 + 5, linspace(0,1), 'red');
        hold on;
    plot(linspace(0,3) * 0 + 10, linspace(0,3), 'red');
        hold on;
    plot(linspace(1,2) * 0 + 6, linspace(1,2), 'red');
        hold on;
    plot(linspace(2,3) * 0 + 7, linspace(2,3), 'red');
        hold on;
    plot(linspace(5,6), linspace(5,6) * 0 + 1, 'red');
        hold on;
    plot(linspace(5,10), linspace(5,10) * 0, 'red');
        hold on;
    plot(linspace(6,7), linspace(6,7) * 0 + 2, 'red');
        hold on;
    plot(linspace(7,10), linspace(7,10) * 0 + 3, 'red');
        hold on;


    
    %Road our vehicle makes 
    plot(x, y,'LineWidth',2,'Color',[0 0.4470 0.7410]);
    hold on;
    %Starting point
    plot(x(1), y(1), 'o','MarkerFaceColor','b')
    hold on;
    %Where we reached
    plot(x(iter-1), y(iter-1),'d','MarkerFaceColor','g'); 
    hold on;
    %Our destination
    plot(destination(1), destination(2), 's','MarkerEdgeColor','r');
    title(['è = ', num2str(angles_init(angles_index)), '°'])
    xlabel('x')
    ylabel('y')
    axis([0 10.1 0 5])
    hold off;

    %Restart for next angle
    iter=2;
    checker=0;
end

saveas(Vehicle, strcat('Vehicle.png'));
close(Vehicle);
