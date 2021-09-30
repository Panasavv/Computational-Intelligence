clear variables;
clc;

%Initiallize
%Import dataset
data = importdata('haberman.data');

%Epochs chosen
epochs = 150;
%Classes
class_no = 2;
%Radius for dependent
clust_rad_dep = [0.15 0.3];
%Radius for independent
clust_rad_ind = [0.25 0.5];


%Split dataset
[train_data, check_data, validate_data] = split_scale(data,1);

%Main

%Dependent
for i = 1:2
    
    %Initiallize error matrix and accuracies
    %we need this here to return to zero in each iteration
    error_matrix = zeros(class_no,class_no);
    suma_of_oa = 0;
    suma_of_k = 0;
    users_accuracy = zeros(class_no,1);
    producers_accuracy = zeros(class_no,1);
    
    %Clustering Per Class
    [c1, sig1] = subclust(train_data(train_data(:, end) == 1, :), clust_rad_dep(i));
    [c2, sig2] = subclust(train_data(train_data(:, end) == 2, :), clust_rad_dep(i));
    rule_no = size(c1, 1) + size(c2, 1);

    %Build a new FIS
    train_fis = newfis('FIS_SC', 'sugeno');

    %Add Input-Output variables
    input_names = {'in1', 'in2', 'in3'};
    for k = 1:(size(train_data, 2) - 1)
        train_fis = addvar(train_fis, 'input', input_names{k}, [0 1]);
    end
    train_fis = addvar(train_fis, 'output', 'out1', [0 1]);

    %Add Input Membership Functions
    for k = 1:size(train_data, 2)-1
        for j = 1:size(c1, 1)
            train_fis = addmf(train_fis, 'input', k, strcat('in',int2str(j)), 'gaussmf', [sig1(k) c1(j, k)]);
        end
        for l = 1:size(c2, 1)
            train_fis = addmf(train_fis, 'input', k,strcat('in',int2str(j+l)), 'gaussmf', [sig2(k) c2(l, k)]);
        end
    end

    %Add Output Membership Functions
    params = [ones(1, size(c1,1)) 2*ones(1, size(c2,1))];
    for k = 1:rule_no
        train_fis = addmf(train_fis, 'output', 1, strcat('out',int2str(k)), 'constant', params(k));
    end

    %Add rules
    ruleList = zeros(rule_no, size(train_data, 2));
    for k = 1:size(ruleList, 1)
        ruleList(k, :) = k;
    end
    ruleList = [ruleList ones(rule_no, 2)];
    train_fis = addrule(train_fis, ruleList);


    %Training and evaluation
    [fis, train_error, steps, checking_fis, checking_error] = ...
        anfis(train_data, train_fis, epochs, NaN, check_data(:,:));

    fis_evaluated = evalfis(validate_data(:,1:end-1), checking_fis);
    fis_evaluated = round(fis_evaluated);

    for j = 1:length(fis_evaluated)
        if(fis_evaluated(j) <= 0)
            fis_evaluated(j) = 1;
        end
        if(fis_evaluated(j) >= 3)
            fis_evaluated(j) = 2;
        end
    end

    %Plotting
    %Membership Function

    membership_function = figure;
    for k = 1:(size(train_data, 2)-1)
        subplot(ceil((size(train_data, 2)-1)/2), 2, k);
        plotmf(checking_fis, 'input', k)
    end
    saveas(membership_function, strcat('membership_functions_dep_',int2str(rule_no), '.png'));
    close(membership_function); 

    %Learning curves
    learning_curve = figure;
    epoch = 1:epochs;
    plot(epoch, train_error .^ 2, 's', epoch, checking_error .^2, '*')
    title('Learning Curves')
    legend('Training Error', 'Validation Error')
    xlabel('Epoch')
    ylabel('MSE')
    saveas(learning_curve, strcat('learning_curves_dep_',int2str(rule_no), '.png'));
    close(learning_curve);

    %Prediction Error
    prediction_error = figure('Position',[0 0 5000 500]);
    plot(1:length(fis_evaluated), fis_evaluated,'Color','b');
    hold on;
    plot(1:length(fis_evaluated), validate_data(:,end),'Color','r');
    title('Predictions')
    legend('Predicted Value', 'Real Value')
    saveas(prediction_error, strcat('prediction_error_dep_',int2str(rule_no), '.png'));           
    close(prediction_error);


    %Error matrix
    for k = 1:class_no
        for j = 1:class_no
            temp = sum((fis_evaluated(:) == k) & (validate_data(:, size(validate_data, 2)) == j));
            error_matrix(k,j)= temp;
        end
        suma_of_oa = suma_of_oa + error_matrix(k,k);              
    end
    
   %Calculate Accuracies
   
   overall_accuracy = suma_of_oa/size(fis_evaluated, 1);
    
   for k = 1:class_no
       suma_of_users(k) = 0;
       suma_of_producers(k) = 0;
       for j = 1:class_no
           suma_of_users(k) = suma_of_users(k) + error_matrix(k,j);
           suma_of_producers(k) = suma_of_producers(k) + error_matrix(j,k);
       end
       suma_of_k = suma_of_k + suma_of_users(k)*suma_of_producers(k); 
       users_accuracy(k) = error_matrix(k,k)/suma_of_users(k);
       producers_accuracy(k) = error_matrix(k,k)/suma_of_producers(k);
   end

   k_hat = (size(fis_evaluated, 1)*suma_of_oa - suma_of_k)/(size(fis_evaluated, 1)^2 - suma_of_k);
    

    
   %Create txts with results
       results = [overall_accuracy k_hat];

       dlmwrite(strcat('OA+k_dep_',int2str(rule_no), '.txt'), results);
       dlmwrite(strcat('producers_accuracy_dep_',int2str(rule_no), '.txt'), producers_accuracy);
       dlmwrite(strcat('users_accuracy_dep_',int2str(rule_no), '.txt'), users_accuracy);
       dlmwrite(strcat('error_matrix_dep_',int2str(rule_no), '.txt'), error_matrix);
end

%Independent
for i = 1:2
    %Initiallize error matrix and accuracies
    %we need this here to return to zero in each iteration
    error_matrix = zeros(class_no,class_no);
    suma_of_oa = 0;
    suma_of_k = 0;
    users_accuracy = zeros(class_no,1);
    producers_accuracy = zeros(class_no,1);
    
    
    %Create Fis model
    train_fis = genfis2(train_data(:,1:end-1), train_data(:,end), clust_rad_ind(i));
    
    %Training and Evaluation
    [fis, train_error, steps, checking_fis, checking_error] = ...
        anfis(train_data, train_fis, epochs, NaN, check_data);
    
    fis_evaluated = evalfis(validate_data(:, 1:end-1), checking_fis);
    fis_evaluated = round(fis_evaluated);
    
    for j = 1:length(fis_evaluated)
        if(fis_evaluated(j) <= 0)
            fis_evaluated(j) = 1;
        end
        if(fis_evaluated(j) >= class_no)
            fis_evaluated(j) = class_no;
        end
    end
    
    %Number of Rules
    rule_no = size(showrule(checking_fis), 1);
    

    %Plotting
    %Membership Function
    
    membership_function = figure;
    for k = 1:(size(train_data, 2)-1)
        subplot(ceil((size(train_data, 2)-1)/2), 2, k);
        plotmf(checking_fis, 'input', k)
    end
    saveas(membership_function, strcat('membership_functions_ind_',int2str(rule_no), '.png'));
    close(membership_function); 
    
    %Learning Curves
    learning_curve = figure;
    epoch = 1:epochs;
    plot(epoch, train_error .^ 2, 's', epoch, checking_error .^2, '*')
    title('Learning Curves')
    legend('Training Error', 'Validation Error')
    xlabel('Epoch')
    ylabel('MSE')

    saveas(learning_curve, strcat('learning_curves_ind_',int2str(rule_no), '.png'));
    close(learning_curve);
    
    %Prediction Error
    prediction_error = figure('Position',[0 0 5000 500]);
    plot(1:length(fis_evaluated), fis_evaluated,'Color','b');
    hold on;
    plot(1:length(fis_evaluated), validate_data(:,end),'Color','r');
    title('Predictions')
    legend('Predicted Value', 'Real Value')
	saveas(prediction_error, strcat('prediction_error_ind_',int2str(rule_no), '.png'));           
    close(prediction_error);

    
    %Error matrix
    for k = 1:class_no
        for j = 1:class_no
            temp = sum((fis_evaluated(:) == k) & (validate_data(:, size(validate_data, 2)) == j));
            error_matrix(k,j)= temp;
        end
        suma_of_oa = suma_of_oa + error_matrix(k,k);              
    end
    
   %Calculate accuracies
   overall_accuracy = trace(error_matrix)/size(fis_evaluated, 1);
   
   for k = 1:class_no
      suma_of_users(k) = 0;
      suma_of_producers(k) = 0;
       for j = 1:class_no
           suma_of_users(k) = suma_of_users(k) + error_matrix(k,j);
           suma_of_producers(k) = suma_of_producers(k) + error_matrix(j,k);
       end
       suma_of_k = suma_of_k + suma_of_users(k)*suma_of_producers(k); 
       users_accuracy(k) = error_matrix(k,k)/suma_of_users(k);
       producers_accuracy(k) = error_matrix(k,k)/suma_of_producers(k);
   end

   k_hat = (size(fis_evaluated, 1)*suma_of_oa - suma_of_k)/(size(fis_evaluated, 1)^2 - suma_of_k);
    

   
    
    %Create txts with results
    results = [overall_accuracy k_hat];

    dlmwrite(strcat('OA+k_ind_',int2str(rule_no), '.txt'), results);
    dlmwrite(strcat('producers_accuracy_ind_',int2str(rule_no), '.txt'), producers_accuracy);
    dlmwrite(strcat('users_accuracy_ind_',int2str(rule_no), '.txt'), users_accuracy);
    dlmwrite(strcat('error_matrix_ind_',int2str(rule_no), '.txt'), error_matrix);
end

