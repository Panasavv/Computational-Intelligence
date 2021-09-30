clear variables;
clc;
%Initiallize
%Import Dataset
data = importdata('airfoil_self_noise.dat');

%Epochs
epochs = 200;
%Output MF type
outMF_type = {'constant'; 'linear'};
%Input MF
inMF_num = [2 3];
%Input MF type
inMF_type = 'gbellmf';
%Split dataset
[train_data, check_data, validate_data] = split_scale(data,1);

%Main
for i = 1:4

        if i==1
            out_type=outMF_type{1};
            in_num=inMF_num(1);
        elseif i==2
            out_type=outMF_type{1};
            in_num=inMF_num(2);
        elseif i==3
            out_type=outMF_type{2};
            in_num=inMF_num(1);
        else
            out_type=outMF_type{2};
            in_num=inMF_num(2);
        end
        
        %Create and train our model
        fis_for_train = genfis1(train_data, in_num, inMF_type, out_type);
         
        [fis, train_error, steps, checking_fis, checking_error] =...
            anfis(train_data, fis_for_train, epochs, NaN, check_data);
        
        %Evaluate our model
        fis_evaluated = evalfis(validate_data(:,1:end-1), checking_fis);
        
        %Plotting
        %Membership functions

        membership_function = figure('Position',[0 0 500 1000]);
        for k=1:5
            subplot(5, 1, k);
            plotmf(checking_fis, 'input', k);
            title(['Membership Function-',out_type,'-',int2str(in_num),'-',int2str(k)]);
            ylabel('Degrees');
        end
        saveas(membership_function, strcat('membership_functions_',int2str(i), '.png'));        
        close(membership_function);

        %Learning curves
        learning_curve = figure;
        epochs2 = 1:epochs;
        plot(epochs2, train_error .^ 2, 's', epochs2,checking_error .^2,'*')
        
        title(['Learning Curve-',out_type,'-',int2str(in_num)]);
        legend('Training Error', 'Validation Error');
        xlabel('Epoch');
        ylabel('Mean Square Error');
        saveas(learning_curve, strcat('learning_curves_',int2str(i),'.png'));
        close(learning_curve);
        
        %Prediction error
        prediction_error = figure('Position', [0 0 6000 450]);
        plot(1:length(fis_evaluated), fis_evaluated,'Color','b');
        hold on;
        plot(1:length(fis_evaluated), validate_data(:,end),'Color','r');      
        title(['Predictions-',out_type,'-',int2str(in_num)]);
        legend('Predicted Value', 'Real Value');
		saveas(prediction_error, strcat('prediction_error_',int2str(i),'.png'));           
        close(prediction_error); 
        
        %Create error txt
        disp([out_type, '_', int2str(in_num)]);
        RMSE = 0;
        for l=1:size(validate_data, 1)
            RMSE = RMSE + (validate_data(l, size(validate_data, 2)) - fis_evaluated(l))^2;
        end

        RMSE = sqrt(RMSE/size(validate_data, 1));
        disp(['RMSE = ' num2str(RMSE)]);
        sy2 = std(validate_data(:, size(validate_data, 2)), 1)^2;
        NMSE = (RMSE^2)/sy2;
        disp(['NMSE = ' num2str(NMSE)]);
        NDEI = sqrt(NMSE);
        disp(['NDEI = ' num2str(NDEI)]);
        SSres = size(data, 1)*(RMSE^2);
        SStot = size(data, 1)*sy2;
        R2 = 1 - SSres/SStot;
        disp(['R^2 = ' num2str(R2)]);
        error_txt = [RMSE NMSE NDEI R2];
        dlmwrite(strcat('errors_',int2str(i),'.txt'),error_txt);
end
