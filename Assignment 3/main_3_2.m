clear variables;
clc;
%Initialize

%Import dataset
data = csvread('train.csv', 1, 0);

%Random selected characteristics
characteristics = [1 5 10 15];
%Rads
clust_rad = [0.55 0.45 0.35 0.25]; 
relief = 25;
%Epochs
epoch = 150;
%Epoch for the pre-processing
epoch_relief = 50; 
%Number of fold cross validation
kfold_no=5; 
grid_score = zeros(length(characteristics), length(clust_rad));
error = zeros(size(characteristics, 2),size(clust_rad, 2),kfold_no);
ticker=0;
RMSE = 0;
disp('Start preprocess');

%Main
%Preproccess our data
[rank, weight] = relieff(data(:, 1:end-1), data(:,end), relief);
[train_data, check_data, valid_data] = split_scale(data, 1);
cross_valid = cvpartition(train_data(:, end), 'KFold', kfold_no);


disp('Start script');
%5-fold cross validation
for char_no = 1:4
    for rad_no = 1:4        
        

        for i = 1:5
            ticker=ticker+1;
			disp(strcat('Iteration number: ',int2str(ticker),' Characteristics-> ',int2str(characteristics(char_no)),...
                ' Rads-> ',num2str(clust_rad(rad_no))));
            training_no = cross_valid.training(i);
            testing_no = cross_valid.test(i);
            
            %Clustering
            cluster_training_data = data(training_no, rank(1:characteristics(char_no)));
			cluster_checking_data = data(testing_no, rank(1:characteristics(char_no)));
			cluster_training_fis = genfis2(cluster_training_data, data(training_no, end),clust_rad(rad_no));

			
			%Training
			[fis, cluster_training_error, cluster_step_size, cluster_checking_fis,cluster_checking_error]=...
                anfis(data(training_no,[rank(1:characteristics(char_no)) end]),cluster_training_fis,...
                    epoch_relief,NaN, data(testing_no, [rank(1:characteristics(char_no)) end]));
            
			%RMSE
			error(char_no,rad_no,i) = min(cluster_checking_error);
        end
		
		grid_score(char_no, rad_no) = sum(error(char_no,rad_no,:))/kfold_no;
		rule_no(char_no,rad_no) = size(showrule(cluster_training_fis), 1); % Number of rules
        
    end
end

[min_column, min_row] = min(grid_score);
[min_score, min_col] = min(min_column);
min_char_no = min_row(min_col);
min_rad_no = min_col;

%Create and train our model
fis_for_train = genfis2(train_data(:,rank(1:characteristics(min_char_no))),...
    train_data(:, end), clust_rad(min_rad_no));

[fis, train_error, steps, checking_fis, checking_error] = ... 
    anfis(train_data(:,[rank(1:characteristics(min_char_no)) end]), fis_for_train,...
        epoch, NaN, check_data(:,[rank(1:characteristics(min_char_no)) end]));

%Then evaluate our model
fis_evaluated = evalfis(valid_data(:, rank(1:characteristics(min_char_no))), checking_fis);




%Plotting
%Membership Function

membership_function = figure;
for j=0:1
	subplot(2, 2, 2*j+1);
	plotmf(fis_for_train, 'input', j+1)
	title('Pre training');
	subplot(2, 2, 2*j+2);
	plotmf(checking_fis, 'input', j+1)
	title('Post Training');
end
	saveas(membership_function, strcat('membership_functions','.png'));        
    close(membership_function);


%Learning curves
learning_curve = figure;
epoch2 = 1:epoch;
plot(epoch2, train_error .^ 2, 's', epoch2, checking_error .^2, '*')
title('Learning Curve')
legend('Training Error', 'Validation Error')
xlabel('Epoch')
ylabel('Mean Square Error')
saveas(learning_curve, strcat('learning_curve', '.png'));
close(learning_curve);

%Prediction error
prediction_error = figure('Position', [0 0 6000 450]);
plot(1:length(fis_evaluated), fis_evaluated,'Color','b');
hold on;
plot(1:length(fis_evaluated), valid_data(:, end),'Color','r');      
title('Predictions')
legend('Predicted Value', 'Real Value')
saveas(prediction_error, strcat('prediction_error_1', '.png'));           
close(prediction_error); 


%Create error txt
for i=1:size(valid_data, 1)
	RMSE = RMSE + (valid_data(i, size(valid_data, 2)) - fis_evaluated(i))^2;
end
RMSE = sqrt(RMSE/size(valid_data, 1));
disp(['RMSE = ' num2str(RMSE)]);
sy2 = std(valid_data(:, size(valid_data, 2)), 1)^2;
NMSE = (RMSE^2)/sy2;
disp(['NMSE = ' num2str(NMSE)]);
NDEI = sqrt(NMSE);
disp(['NDEI = ' num2str(NDEI)]);
SSres = size(data, 1)*(RMSE^2);
SStot = size(data, 1)*sy2;
R2 = 1 - SSres/SStot;
disp(['R^2 = ' num2str(R2)]);
coefERR = [RMSE NMSE NDEI R2];
dlmwrite(strcat('errors.txt'), coefERR);


