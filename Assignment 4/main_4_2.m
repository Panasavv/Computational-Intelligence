clear variables;
clc;

%Initialize
%Import dataset.
data = csvread('data.csv', 1, 1);

%Random Selected characteristics
characteristics = [3 6 12 15];
%Rads
clust_rad = [0.5 0.35 0.2 0.05];
relief = 175;
%Epochs
epochs = 150;
%Epoch for the pre-processing
epoch_relief = 50;
classes_no = 5;
grid_score = zeros(length(characteristics), length(clust_rad));
iteration_number=0;

error_matrix = zeros(classes_no, classes_no);
suma_of_oa = 0;
suma_of_k = 0;
users_accuracy = zeros(classes_no,1);
producers_accuracy = zeros(classes_no,1);


disp('start script');

%Main
%Preproccess our data
[rank, weight] = relieff(data(:, 1:end-1), data(:,end), relief);

[train_data,check_data,valid_data]=split_scale(data,1);
cross_valid = cvpartition(train_data(:, end), 'KFold', 5, 'Stratify', true);

%5-fold cross validation
for char_no = 1:4
    for rad_no = 1:4

        suma_for_error=0;
        for i = 1:cross_valid.NumTestSets
            iteration_number=iteration_number+1;
            disp(strcat('Iteration number: ',int2str(iteration_number),' char-> ',int2str(characteristics(char_no)),...
                ' rad-> ',num2str(clust_rad(rad_no))));
            train_no = cross_valid.training(i);
            check_no = cross_valid.test(i);
            
            %Create FIS object
            cluster_training_data = data(train_no, rank(1:characteristics(char_no)));
			cluster_checking_data = data(check_no, rank(1:characteristics(char_no)));            
                    
            [c1, sig1] = subclust(cluster_training_data(cluster_training_data(:, end) == 1, :), clust_rad(rad_no));
            [c2, sig2] = subclust(cluster_training_data(cluster_training_data(:, end) == 2, :), clust_rad(rad_no));
            [c3, sig3] = subclust(cluster_training_data(cluster_training_data(:, end) == 3, :), clust_rad(rad_no));
            [c4, sig4] = subclust(cluster_training_data(cluster_training_data(:, end) == 4, :), clust_rad(rad_no));
            [c5, sig5] = subclust(cluster_training_data(cluster_training_data(:, end) == 5, :), clust_rad(rad_no));
        
            rule_no = size(c1, 1) + size(c2, 1) + size(c3, 1) + size(c4, 1) + size(c5, 1);       
            
            %Build a new FIS
            train_fis = newfis('FIS_SC', 'sugeno');
            
            %Add Input-Output variables
            for I = 1:(size(cluster_training_data, 2) - 1)
                in{I} = {['in' num2str(I)]};
            end
            for k = 1:(size(cluster_training_data, 2)-1)
                %in{k} = {['in' num2str(k)]};
                train_fis = addvar(train_fis, 'input', in{k}, [0 1]);
            end
            train_fis = addvar(train_fis, 'output', 'out1', [0 1]);       
            
            %Add Input Membership Functions
            name='sth';
            for k = 1:size(cluster_training_data, 2)-1
                for j = 1:size(c1, 1)
                    train_fis = addmf(train_fis, 'input', k, name, 'gaussmf', [sig1(k) c1(j, k)]);
                end
                for j = 1:size(c2, 1)
                    train_fis = addmf(train_fis, 'input', k, name,'gaussmf', [sig2(k) c2(j, k)]);
                end
                for j = 1:size(c3, 1)
                    train_fis = addmf(train_fis, 'input', k, name, 'gaussmf', [sig3(k) c3(j, k)]);
                end
                for j = 1:size(c4, 1)
                    train_fis = addmf(train_fis, 'input', k, name, 'gaussmf', [sig4(k) c4(j, k)]);
                end
                for j = 1:size(c5, 1)
                    train_fis = addmf(train_fis, 'input', k, name, 'gaussmf', [sig5(k) c5(j, k)]);
                end
            end
            
            %Add Output Membership Functions
            params = [ones(1, size(c1,1)) 2*ones(1, size(c2,1))...
                3*ones(1, size(c3,1)) 4*ones(1, size(c4,1)) 5*ones(1, size(c5,1))];
            for k = 1:rule_no
                train_fis = addmf(train_fis, 'output', 1, name,'constant', params(k));
            end
            
            %Add rules
            ruleList = zeros(rule_no, size(cluster_training_data, 2));
            for k = 1:size(ruleList, 1)
                ruleList(k, :) = k;
            end
            ruleList = [ruleList ones(rule_no, 2)];
            train_fis = addrule(train_fis, ruleList);
            
			%Training and evaluation
			[fis, train_error, steps, checking_fis, checking_error] ...
                = anfis(data(train_no, [rank(1:characteristics(char_no)) end]), train_fis, epoch_relief, ...
                    NaN, data(check_no, [rank(1:characteristics(char_no)) end]));
  
            
            if ~isnan(checking_error(epoch_relief))
                suma_for_error = suma_for_error + checking_error(epoch_relief);
            end
			
        end
            
        grid_score(char_no, rad_no) = suma_for_error / cross_valid.NumTestSets;
        rules(char_no,rad_no) = rule_no;
           
    end
end
            
[min_columns, min_row] = min(grid_score);
[min_score, min_col] = min(min_columns);
min_char_no = min_row(min_col);
min_rad_no = min_col;

%Create FIS object
train_fis = genfis2(train_data(:,rank(1:characteristics(min_char_no))), train_data(:, end),...
    clust_rad(min_rad_no));

%Training and evaluation
[fis, train_error, steps, checking_fis, checking_error] = ... 
    anfis(train_data(:,[rank(1:characteristics(min_char_no)) end]), train_fis, epochs,...
        NaN, check_data(:,[rank(1:characteristics(min_char_no)) end]));

fis_evaluated = evalfis(valid_data(:, rank(1:characteristics(min_char_no))), checking_fis);

%Round the results for better outcomes
fis_evaluated = round(fis_evaluated);

for j = 1:length(fis_evaluated)
    if (fis_evaluated(j) <= 0)
        fis_evaluated(j) = 1;
    end
    if(fis_evaluated(j) >= classes_no)
        fis_evaluated(j) = 5;
    end
end


%Plotting
%Membership Function

membership_function = figure;
for j = 0:1
    subplot(2, 2, 2*j+1);
    plotmf(train_fis, 'input', j+1)
    title('Pre training');
    subplot(2, 2, 2*j+2);
	plotmf(checking_fis, 'input', j+1)
    title('Post Training');
end
saveas(membership_function, strcat('membership_functions_',int2str(rule_no), '.png'));
close(membership_function); 

%Learning Curves
learning_curves = figure;
epoch = 1:epochs;
plot(epoch, train_error .^ 2, 's', epoch, checking_error .^2, '*')
title('Learning Curves')
legend('Training Error', 'Validation Error')
xlabel('Epoch')
ylabel('Mean Square Error')
saveas(learning_curves, strcat('learning_curve_',int2str(rule_no), '.png'));
close(learning_curves);

%Prediction error
prediction_error = figure('Position',[0 0 5000 500]);
plot(1:length(fis_evaluated), fis_evaluated); hold on;
plot(1:length(fis_evaluated), valid_data(:,end));
title('Value Predictions')
legend('Predicted Value', 'Real Value')
saveas(prediction_error, strcat('prediction_error_',int2str(rule_no), '.png'));           
close(prediction_error);


%Error matrix
for k = 1:classes_no
    for j = 1:classes_no
        temp = sum((fis_evaluated(:) == k) & (valid_data(:, size(valid_data, 2)) == j));
        error_matrix(k,j)= temp;
    end
    suma_of_oa = suma_of_oa + error_matrix(k,k);              
end

%Calculate Accuracies
overall_accuracy = suma_of_oa/size(fis_evaluated, 1);

for k = 1:classes_no
    suma_of_users(k) = 0;
    suma_of_producers(k) = 0;
    for j = 1:classes_no
        suma_of_users(k) = suma_of_users(k) + error_matrix(k,j);
        suma_of_producers(k) = suma_of_producers(k) + error_matrix(j,k);           
    end
    suma_of_k = suma_of_k + suma_of_users(k)*suma_of_producers(k);
    users_accuracy(k) = error_matrix(k,k)/suma_of_users(k);
    producers_accuracy(k) = error_matrix(k,k)/suma_of_producers(k);
end

k_hat = (size(fis_evaluated, 1)*suma_of_oa - suma_of_k)/(size(fis_evaluated, 1)^2 - suma_of_k);

%Create error txts
results = [overall_accuracy k_hat];

dlmwrite(strcat('OA+k_',int2str(rule_no), '.txt'), results);
dlmwrite(strcat('producer_accuracy_', int2str(rule_no), '.txt'), producers_accuracy);
dlmwrite(strcat('users_accuracy_',int2str(rule_no), '.txt'), users_accuracy);
dlmwrite(strcat('error_matrix_',int2str(rule_no), '.txt'), error_matrix);    