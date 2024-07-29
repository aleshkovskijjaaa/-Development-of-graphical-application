% function pattern_reco_lab1_mod5 %(C, N, theta, centers, sigmas, p_w, axis_rect)

global N_data nTrain_data;
global C data_type_per_class is_null_class ;
global data_type class_data;
global p_w C_matr bayes_type n_class_2 pos_class theta;
global classType;
global isNaiveBayes_data normType_data;
global parDistrType_data nCompPar_data gmRegularize_data sharedCov_data replicates_data algStart_data covType_data;
global parzenType_data smoothPar_data smoothParCalc_data;
global axis_rect deltaT2 deltaT3 Np_calc_length k_curve;
global kNN_K kNN_MetricType kNN_TieAlgType kNN_kw kNN_kw_CalcAlg kNN_Pmink;
global SVM_KernelType SVM_SolverType SVM_C SVM_outlier_fr SVM_pol_degree SVM_MLP_params SVM_auto_scale SVM_scale_value;
global NN_arch_type NN_out_type NN_hidden_type NN_perf_type NN_max_fail NN_train_alg NN_hidden_neurons;
global dataPR dataPR2;
global Tree_MaxNumSplits Tree_MinLeafSize Tree_MinParentSize Tree_MergeLeaves;

% Данные классов
% class_data = SetDataVariants;
class_data = SetDataVariants;
% Тип классификатора 1 - Байесовский, 2 - Параметрический, 3 - KDE, 4 -
% k-NN, 5 - SVM, 6 - НС, 7 - Tree
classType = 2;
% Параметричекое распределение - Нормальное (1), Смесь GMM (2)
parDistrType_data  = {1, 2, 2*ones(1,C(data_type))};
% Количество компонент в смеси для парам. оценки GMM
nCompPar_data  = {1, 4, 4*ones(1,C(data_type))};
gmRegularize_data = {1, 0.1, 0.1*ones(1,C(data_type))};
sharedCov_data = {1, 0, 0*ones(1,C(data_type))};
replicates_data = {1, 1, 1*ones(1,C(data_type))};
algStart_data = {1, 1, 1*ones(1,C(data_type))};
covType_data = {1, 1, 1*ones(1,C(data_type))};


% Наивный (1) или нет (2) оцениватель - если да, то распределения по признакам независимы
isNaiveBayes_data =  {1, 2, 2*ones(1,C(data_type))};
% Тип нормализации
normType_data = {1, 1, ones(1,C(data_type))};
% Тип парзеновских окон
parzenType_data = {1, [1;1], [1;1]*ones(1,C(data_type))};
% Сглаживающие параметры (ширина окна)
smoothPar_data = {1, [.1;.2], [.1;.2]*ones(1,C(data_type))};
% Параметры алгоритма нахождения сглаживающих параметров
smoothParCalc_data = {1, [3;1], [1;1]*ones(1,C(data_type)+1)};
% Параметры алгоритма k ближайших соседей
kNN_K = 3;          % Количество ближайших соседей
kNN_MetricType = 2; % Метрика
kNN_TieAlgType = 1; % Алгоритм разрешения консенсусов
kNN_kw = 1;         % Отношение весов w2/w1
kNN_kw_CalcAlg = 1; % Алгоритм подбора весов
kNN_Pmink = 2;      % Степень в метрике Минковского
% Параметры SVM-классификатора
SVM_KernelType = 3;
SVM_SolverType = 1;
SVM_C = 1;
SVM_pol_degree = 3;
SVM_MLP_params = [1 -1];
SVM_auto_scale = 0;
SVM_scale_value = 1;
SVM_outlier_fr = 0.05;
% Параметры нейронной сети
NN_hidden_type = 1;
NN_perf_type = 2;
NN_max_fail = 10;
NN_arch_type = 1;
NN_out_type = 1;
NN_train_alg = 10;
NN_hidden_neurons = 100;
% Параметры бинарного дерева принятия решений
Tree_MaxNumSplits = 100;
Tree_MinLeafSize = 1;
Tree_MinParentSize = 10;
Tree_MergeLeaves = 1;

h_dlg_set_param  = [];

% Параметры исходных данных

% Объем выборки для одного класса
N_data{1} = 1;
N_data{2} = 1e4;
N_data{3} = N_data{2}*ones(1, C(data_type));
% Объем обучающей выборки
nTrain_data{1} = 1;
nTrain_data{2} = 2e2;
nTrain_data{3} = nTrain_data{2}*ones(1, C(data_type));
n_class_2 = [2 3]; % Какие классы использовать в примере с ROC-кривой
pos_class = 1;  % Какой класс положительный (1 или 2)

%C = [3 3 3];      % Количество классов
%data_type_per_class = [1 1 1; 2 2 2; 3 3 3];


% Параметры классификатора
bayes_type = 1;         % Тип Байесова классификатора (1-ML, 2-MAP, 3 - обобщенный)
C_matr = [              % Матрица стоимости C[i,j] - стоимость классифицировать i класс как j
    0 1 1;
    200 0 200;
    15 15 0 ];
C_matr = 1 - eye(C(data_type),C(data_type));
p_w = [.33 .33 .34]';       % Априорные вероятности (для MAP-классификатора)
p_w = 1/C(data_type)*ones(C(data_type),1);
theta = 1e-5;              % Порог отклонения (значение правдоподобия, ниже которого пример не классифицируется

% Параметры построения графиков
deltaT2 = [5 5];              % Шаг при построении графика типа T2
deltaT3 = 5;
deltaT0 = 5;

Np_calc_length = 200;
k_curve = 1;

colors = {[1 0 0], [0 1 0], [ 0 0 1], [1 1 0], [1 0 1], [0 1 1], 1/255*[255 165 0], 1/255*[47 79 79], 1/255*[188 143 143], 1/255*[139 69 19], 1/255*[0 191 255] };  % Цвета для каждого класса
markertypes = { 'x', 'o', '.', 'diamond','x', 'o', '.', 'diamond', 'x', 'o', '.', 'diamond' };    % Типы маркеров для каждого класса

close all;
% Первичная генерация данных
params.net_need_rebuild = 1; params.net_need_retrain = 1; params.svm_need_retrain = 1; params.tree_need_retrain = 1;
learners.net = create_neural_network(NN_arch_type, NN_out_type, NN_hidden_type, NN_hidden_neurons); learners.svms = {}; learners.tree = [];
[dataPR.dataAll, dataPR.groupsAllReal] = GenerateData(data_type, get_N(N_data, data_type), 1:C(data_type));
[dataPR.dataTrain, dataPR.groupsTrain] = GenerateData(data_type, get_N(nTrain_data, data_type), 1:C(data_type));
dataPR.params = params;
dataPR.learners = learners;
dataPR.params.net_need_rebuild = 0;

[dataPR2.dataAll, dataPR2.groupsAllReal] = GenerateData(data_type, get_N(N_data,data_type), n_class_2);
[dataPR2.dataTrain, dataPR2.groupsTrain] = GenerateData(data_type, get_N(nTrain_data,data_type), n_class_2);
dataPR2.params = params; 
dataPR2.learners = learners;
dataPR2.params.net_need_rebuild = 0;

h_caption = 'What to do';
h_questions = {'Plot data points', 'Plot pdf', 'Plot T1 + Error calc', 'Plot T2', 'ROC', 'Regenerate Data', 'Change Parameters', 'Close all', 'Exit'};
h = menu(h_caption,h_questions );

while isempty(find(h == [0 length(h_questions)],1))
    switch h
        case 1
            % Построение исходных данных
            h2 = menu('Plot data points', 'Only points', '+True Levels', '+Est Levels', '+True, Est Levels', 'Train Sample', 'Back');
            
            while isempty(find(h2 == [0 6],1))
                figure;
                hold on; grid on;
                for c = 1:C(data_type)
                    if h2 == 5
                        plot(dataPR.dataTrain(dataPR.groupsTrain==c,1), dataPR.dataTrain(dataPR.groupsTrain==c,2), 'linestyle','none', 'color',colors{c}, 'marker', markertypes{c});
                    else
                        plot(dataPR.dataAll(dataPR.groupsAllReal==c,1), dataPR.dataAll(dataPR.groupsAllReal==c,2), 'linestyle','none', 'color',colors{c}, 'marker', markertypes{c});
                    end
                end
                axis(axis_rect{data_type});
                legend( get_legend(h, h2));
                title([GetCurrentExpParamsStr(h, h2) ]);
                
                if ~isempty(find(h2==[2 3 4], 1))
                    x1 = axis_rect{data_type}(1):deltaT0:axis_rect{data_type}(2);
                    x2 = axis_rect{data_type}(3):deltaT0:axis_rect{data_type}(4);
                    [X1,X2] = meshgrid(x1,x2);
                    X = [X1(:) X2(:)];
                    
                    [ZZ1, c_levels1, dataPR.learners, dataPR.params] = get_density(X, dataPR.dataTrain, dataPR.groupsTrain, 1, 1:C(data_type), dataPR.learners, dataPR.params);

                    is_error = 1;
%                    try
                        [ZZ2, c_levels2, dataPR.learners, dataPR.params] = get_density(X, dataPR.dataTrain, dataPR.groupsTrain, classType, 1:C(data_type),dataPR.learners, dataPR.params);
                        is_error = 0;
%                    catch ME
%                        h = errordlg(sprintf('%s:\r\nIncrease number of examples', ME.message),sprintf('Error: %s', ME.identifier), 'modal');
%                    end

                    for c = 1:C(data_type)
                        Z1 = ZZ1(c,:)';
                        
                        Z1 = reshape(Z1,length(x2), length(x1));
                        color_comp = colors{c}*.7;
                        if ~isempty(find(h2 == [2 4], 1))
                            if data_type_per_class{data_type}(c) == 3
                                contour(X1, X2, Z1, 'color' , color_comp, 'linewidth', 2);
                            else
                                contour(X1, X2, Z1, [c_levels1(c) c_levels1(c)], 'color' , color_comp, 'linewidth', 2);
                            end
                        end
                        if ~isempty(find(h2 == [3 4], 1)) && isempty(find(classType == [1 5],1)) && ~is_error
                            Z2 = ZZ2(c,:)';
                            Z2 = reshape(Z2,length(x2), length(x1));
                            contour(X1, X2, Z2, [c_levels2(c) c_levels2(c)], 'color' , color_comp, 'linewidth', 1.5, 'linestyle', '--');
                        end
                    end
                end
            h2 = menu('Plot data points', 'Only points', '+True Levels', '+Est Levels', '+True, Est Levels', 'Train Sample', 'Back');
            end
        case 2
            hNew = menu('Choose Plot Type', 'X1,X2', 'X1', 'X2', 'Back');
            
            while isempty(find(hNew == [0 4],1))
                % Построение поверхностей плотностей
                h2 = menu('Plot pdf', 'Estimation PDF', 'True pdf', 'Both pdfs', 'Mean Error', 'Change params', 'Back');

                while isempty(find(h2 == [0 6],1))
                    x1 = axis_rect{data_type}(1):deltaT3:axis_rect{data_type}(2);
                    x2 = axis_rect{data_type}(3):deltaT3:axis_rect{data_type}(4);
                    [X1,X2] = meshgrid(x1,x2);
                    X = [X1(:) X2(:)];

                    n_cmap_levels = 32;
                    cmap_all = [];
                    for c = 1:C(data_type)
                        cmaps{c} = colors{c};
                        cmap_all = [cmap_all; cmaps{c}];
                    end
                    is_error = 0;
                    if ~isempty(find(h2 == [1 3 4], 1))
                        is_error = 1;
    %                    try
                            [ZZ1,~,dataPR.learners, dataPR.params] = get_density(X, dataPR.dataTrain, dataPR.groupsTrain, classType, 1:C(data_type),dataPR.learners, dataPR.params); ZZ = ZZ1;
                            is_error = 0;
    %                    catch ME
    %                        ZZ1 = [];
    %                        h = errordlg(sprintf('%s:\r\nIncrease number of examples', ME.message),sprintf('Error: %s', ME.identifier), 'modal');
    %                    end
                    end
                    if ~isempty(find(h2 == [2 3 4], 1))
                        [ZZ2,~, dataPR.learners, dataPR.params] = get_density(X, dataPR.dataTrain, dataPR.groupsTrain, 1, 1:C(data_type),dataPR.learners, dataPR.params); ZZ = ZZ2;
                    end

%                     Z1_X = {}
%                     Z2_Y = {}
%                     for c = 1:C(data_type)
%                         Z = ZZ(c,:)';
%                         Z = reshape(Z,length(x2), length(x1));
%                         Z_X{c} = sum(Z,1);
%                         Z_Y{c} = sum(Z,2);
%                     end


                    titles = {'Estimation PDF', 'Real PDF'};

                    if ~isempty(find(h2 == [1 2], 1)) && is_error==0
                        menu_pps = {};
                        menu_pps{1} = 'For all classes';
                        c_diaps = {1:C(data_type)};
                        for i = 1:C(data_type)
                            menu_pps{i+1} = ['Class ' num2str(i)];
                            c_diaps{i+1} = i;
                        end
                        menu_pps{C(data_type)+2} = 'Back';

                        h3 = menu( titles{h2}, menu_pps);
                        while isempty(find(h3 == [0 C(data_type)+2],1))
                            if hNew == 1
                                figure; hold on; grid on; view(3); colormap(cmap_all(c_diaps{h3},:));
                                for i = 1:length(c_diaps{h3})
                                    c = c_diaps{h3}(i);
                                    Z = ZZ(c,:)';
                                    Z = reshape(Z,length(x2), length(x1));
                                    hh(c) = surf(X1,X2,Z, 'CData', c*ones(size(Z)), 'EdgeAlpha', .3 );
                                end
                            else
                                figure; hold on; grid on;
                                for i = 1:length(c_diaps{h3})
                                    c = c_diaps{h3}(i);
                                    Z = ZZ(c,:)';
                                    Z = reshape(Z,length(x2), length(x1));
                                    outData = GetZ1d(hNew, Z, x1, x2);
                                    xd = outData{1}; Z1d = outData{2}; x_label = outData{3};

                                    plot(xd,Z1d,'color',colors{c});
                                    xlabel(x_label)
                                end
                            end
                            legend( get_legend(h, h2));
                            title([GetCurrentExpParamsStr(h, h2) ]);

                            h3 = menu( titles{h2}, menu_pps);
                        end
                    elseif h2 == 3
                        menu_pps = {};
                        menu_pps{1} = 'For all classes';
                        c_diaps = {1:C(data_type)};
                        for i = 1:C(data_type)
                            menu_pps{i+1} = ['Class ' num2str(i)];
                            c_diaps{i+1} = i;
                        end
                        menu_pps{C(data_type)+2} = 'Back';

                        h3 = menu( 'Both pdfs', menu_pps);
                        while isempty(find(h3 == [0 C(data_type)+2],1))
                            n_plots = [2 1 1; 2 1 2];
                            Zs = {ZZ1, ZZ2};
                            if hNew == 1
                                figure; hold on; grid on; view(3); colormap(cmap_all(c_diaps{h3},:));
                                for i = 1:2
                                    subplot(n_plots(i,1), n_plots(i,2), n_plots(i,3));
                                    hold on; grid on; view(3);
                                    title(titles{i});

                                    for k = 1:length(c_diaps{h3})
                                        c = c_diaps{h3}(k);
                                        if i == 1 && is_error~=0
                                            continue;
                                        end
                                        Z = Zs{i}(c,:)';
                                        Z = reshape(Z,length(x2), length(x1));
                                        hh(c) = surf(X1,X2,Z, 'CData', c*ones(size(Z)), 'EdgeAlpha', .3 );
                                    end
                                end
                            else
                                figure; hold on; grid on;
                                for i = 1:2
                                    subplot(n_plots(i,1), n_plots(i,2), n_plots(i,3));
                                    hold on; grid on;
                                    title(titles{i});

                                    for k = 1:length(c_diaps{h3})
                                        c = c_diaps{h3}(k);
                                        if i == 1 && is_error~=0
                                            continue;
                                        end
                                        Z = Zs{i}(c,:)';
                                        Z = reshape(Z,length(x2), length(x1));

                                        outData = GetZ1d(hNew, Z, x1, x2);
                                        xd = outData{1}; Z1d = outData{2}; x_label = outData{3};

                                        plot(xd,Z1d,'color',colors{c});
                                        xlabel(x_label)
                                    end
                                end
                            end
                            h3 = menu( 'Both pdfs', menu_pps);
                        end
                    elseif h2 == 4 && is_error == 0
                        E = abs(ZZ1-ZZ2);
                        norm(E)
                        menu_pps = {};
                        for i = 1:C(data_type)
                            menu_pps{i} = ['Class ' num2str(i)];
                        end
                        menu_pps{C(data_type)+1} = 'For all classes separately';
                        menu_pps{C(data_type)+2} = 'For all classes mean';
                        menu_pps{C(data_type)+3} = 'Back';

                        h3 = menu('Mean error', menu_pps);

                        while isempty(find(h3 == [0 C(data_type)+3],1))
                            if hNew == 1
                                figure; hold on; grid on; view(3); 
                                if h3 <= C(data_type)
                                    E_plot = E(h3,:)';
                                elseif h3 == C(data_type)+2
                                    E_plot = mean(E);
                                end
                                if h3 <= C(data_type) || h3 == C(data_type)+2
                                    colormap(1/255*[255 165 0]); % orange
                                    Z = reshape(E_plot,length(x2), length(x1));
                                    norm(Z)
                                    sum(E)/sum(ZZ2)*100
                                    h_e = surf(X1,X2,Z, 'EdgeAlpha', .3 );
                                else
                                    colormap(cmap_all);
                                    for c = 1:C(data_type)
                                        Z = E(c,:)';
                                        Z = reshape(Z,length(x2), length(x1));
                                        h_ee(c) = surf(X1,X2,Z, 'CData', c*ones(size(Z)), 'EdgeAlpha', .3 );
                                    end
                                    legend( get_legend(h, h2));
                                end
                            else
                                figure; hold on; grid on;
                                if h3 <= C(data_type)
                                    E_plot = E(h3,:)';
                                elseif h3 == C(data_type)+2
                                    E_plot = mean(E);
                                end
                                if h3 <= C(data_type) || h3 == C(data_type)+2
                                    colormap(1/255*[255 165 0]); % orange
                                    Z = reshape(E_plot,length(x2), length(x1));
                                    norm(Z)
                                    sum(E)/sum(ZZ2)*100

                                    outData = GetZ1d(hNew, Z, x1, x2);
                                    xd = outData{1}; Z1d = outData{2}; x_label = outData{3};
                                    plot(xd,Z1d,'color',colors{c});
                                    xlabel(x_label)
                                else
                                    colormap(cmap_all);
                                    for c = 1:C(data_type)
                                        Z = E(c,:)';
                                        Z = reshape(Z,length(x2), length(x1));

                                        outData = GetZ1d(hNew, Z, x1, x2);
                                        xd = outData{1}; Z1d = outData{2}; x_label = outData{3};
                                        plot(xd,Z1d,'color',colors{c});
                                        xlabel(x_label)
                                    end
                                    legend( get_legend(h, h2));
                                end
                            end
                            h3 = menu('Mean error', menu_pps);
                        end
                    elseif h2 == 5 && is_error == 0
                        prompt_basic = {'Enter deltaT3 value:','Enter Np_calc_length value:', 'Enter k_curve value:'};
                        prompt = prompt_basic;
                        dlg_title = 'Value:';
                        num_lines = 1;
                        defaultans = {num2str(deltaT3),num2str(Np_calc_length), num2str(k_curve)};
                        dataT = [deltaT3, Np_calc_length, k_curve];
                        while 1
                            answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
                            if isempty(answer)
                                break;
                            end
                            isError = 0;
                            for i = 1:3
                                defaultans{i} = answer{i};
                                valT = str2num(defaultans{i});
                                if isempty(valT)
                                    isError = 1;
                                elseif ~isscalar(valT)
                                    isError = 1;
                                    continue;
                                else
                                    prompt{i} = prompt_basic{i}
                                    dataT(i) = valT;
                                end
                                if isError ~= 0
                                    prompt{i} =  ['Error! ' prompt_basic{i}]
                                end
                            end
                            if isError == 0
                                deltaT3 = dataT(1);
                                Np_calc_length = dataT(2);
                                k_curve = dataT(3);
                                break;
                            end
                        end
                        
                        %            axis(axis_rect{data_type});
                    end
                    h2 = menu('Plot pdf', 'Estimation PDF', 'True pdf', 'Both pdfs', 'Mean Error', 'Change params', 'Back');
                end
                hNew = menu('Choose Plot Type', 'X1,X2', 'X1', 'X2', 'Back')
            end
        case 3
            % График типа T1 и расчет ошибок
            is_error = 1;
%            try
                [dataPR.groupsAllResult, dataPR.learners, dataPR.params] = classify_bayes_all(dataPR.dataAll, dataPR.dataTrain, dataPR.groupsTrain, dataPR.learners, dataPR.params);
                is_error = 0;
%            catch ME
%                ZZ1 = [];
%                h = errordlg(sprintf('%s:\r\nIncrease number of examples', ME.message),sprintf('Error: %s', ME.identifier), 'modal');
%            end
            
            if is_error == 0
                % Средняя ошибка
                mean_error = sum(dataPR.groupsAllResult~=dataPR.groupsAllReal)/length(dataPR.groupsAllReal);
                % Матрица ошибок
                confmatr = confusionmat(dataPR.groupsAllReal, dataPR.groupsAllResult)
                % Ошибки 1 рода (ложное срабатывание) и 2 рода (пропуск события)
                p_e_1 = zeros(C(data_type),1);
                p_e_2 = zeros(C(data_type),1);
                % Учет нулевого класса (отклоненные примеры)
                if size(confmatr,1) == C(data_type)+1
                    ddd = 1;
                else
                    ddd = 0;
                end
                for i = 1:C(data_type)
                    ii = i + ddd;
                    p_e_1(i) = sum(confmatr([1:C(data_type)+ddd]~=ii, ii)) / sum(confmatr(1:C(data_type)+ddd,ii));
                    p_e_2(i) = sum(confmatr(ii,[1:C(data_type)+ddd]~=ii)) / sum(confmatr(ii,1:C(data_type)+ddd));
                end
                
                mean_error
                p_e_1
                p_e_2
                
                if ~isempty(find(dataPR.groupsAllResult==0))
                    is_null_class = 1;
                else
                    is_null_class = 0;
                end
                
                % Построение
                figure;
                hold on; grid on;
                for c = 1:C(data_type)
                    plot(dataPR.dataAll(dataPR.groupsAllResult==c,1), dataPR.dataAll(dataPR.groupsAllResult==c,2), 'linestyle','none', 'color',colors{c}, 'marker', markertypes{c});
                end
                plot(dataPR.dataAll(dataPR.groupsAllResult==0,1), dataPR.dataAll(dataPR.groupsAllResult==0,2), 'linestyle','none', 'color', 'black', 'marker', 'square');
                
                if classType == 5
                    for c = 1:C(data_type)
                        supVecC = dataPR.dataTrain(dataPR.learners.svms{c}.IsSupportVector,:);
                        plot(supVecC(:,1),supVecC(:,2), 'linestyle','none', 'color', colors{c}, 'marker', 'o', 'markeredgecolor', 'black', 'linewidth',1.5, 'markersize', 6);
                    end
                    legend_extra = 'Support Vectors';
                else
                    legend_extra = [];
                end
                
                axis(axis_rect{data_type});
                legend( [get_legend(h) legend_extra] );
                title([GetCurrentExpParamsStr(h, 1) ]);
            end
        case 4
            % График типа T2
            % Расчет
            
            h2 = menu('Plot T2', 'Plot', 'Change params', 'Back');
            
            while isempty(find(h2 == [0 3],1))
                switch h2
                    case 1
                        x1 = axis_rect{data_type}(1):deltaT2(1):axis_rect{data_type}(2);
                        x2 = axis_rect{data_type}(3):deltaT2(2):axis_rect{data_type}(4);
                        [X1,X2] = meshgrid(x1,x2);
                        X = [X1(:) X2(:)];
                        
                        is_error = 1;
%                        try
                            [Z, dataPR.learners, dataPR.params] = classify_bayes_all(X, dataPR.dataTrain, dataPR.groupsTrain, dataPR.learners, dataPR.params);
                            is_error = 0;
%                        catch ME
%                            ZZ1 = [];
%                            h = errordlg(sprintf('%s:\r\nIncrease number of examples', ME.message),sprintf('Error: %s', ME.identifier), 'modal');
%                        end
                        
                        if is_error == 0
                            Z = reshape(Z,length(x2), length(x1));
                            
                            % Определение, было ли отклонение по каким-либо примером
                            if ~isempty(find(Z==0))
                                is_null_class = 1;
                            else
                                is_null_class = 0;
                            end
                            
                            figure;
                            hold on;
                            % Установка цветов в зависимости от того, было отклонение или
                            % нет
                            rejected_color = .8*[1 1 1];
                            
                            cmap_cur = [];
                            for i = 1:C(data_type)
                                cmap_cur = [cmap_cur; colors{i}];
                            end
                            
                            if is_null_class
                                cmap_cur = [rejected_color;cmap_cur];
                                i_scatter = [2:size(cmap_cur,1) 1];
                            else
                                i_scatter = 1:size(cmap_cur,1);
                            end
                            
                            for i = i_scatter
                                scatter(0,0,1,cmap_cur(i,:),'filled');
                            end
                            
                            % Построение
                            %fill(X1,X2,Z);
                            colormap(cmap_cur);
                            surf(X1,X2,Z, 'edgecolor','none', 'CData', Z+1 );
                            view(2)
                            axis(axis_rect{data_type});
                            grid on;
                            
                            legend( get_legend(h));
                            title([GetCurrentExpParamsStr(h, 1) ]);
                        end
                    case 2
                        prompt = {'Enter deltaX1 value:','Enter deltaX2 value:'};
                        dlg_title = 'Value:';
                        num_lines = 1;
                        defaultans = {num2str(deltaT2(1)),num2str(deltaT2(2))};
                        deltaT2t = deltaT2;
                        while 1
                            answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
                            if isempty(answer)
                                return;
                            end
                            isError = 0;
                            for i = 1:2
                                defaultans{i} = answer{i};
                                valT = str2num(defaultans{i});
                                if isempty(valT)
                                    isError = 1;
                                elseif ~isscalar(valT)
                                    isError = 1;
                                    continue;
                                else
                                    prompt{i} = sprintf('Enter deltaX%d value:', i);
                                    deltaT2t(i) = valT;
                                end
                                if isError ~= 0
                                    prompt{i} = sprintf('Error! Enter correct deltaX%d value:', i);
                                end
                            end
                            if isError == 0
                                deltaT2(1) = deltaT2t(1);
                                deltaT2(2) = deltaT2t(2);
                                break;
                            end
                        end
                        
                end
                h2 = menu('Plot T2', 'Plot', 'Change params', 'Back');
            end

        case 5
            % Расчет ROC-кривой и порогов для различных критериев
            is_error = 1;
%            try
                [dataPR2.scores, dataPR2.learners, dataPR2.params] = classify_bayes_all_soft(dataPR2.dataAll, dataPR2.groupsAllReal, dataPR2.dataTrain, dataPR2.groupsTrain, dataPR2.learners, dataPR2.params);
                is_error = 0;
%            catch ME
%                ZZ1 = [];
%                h = errordlg(sprintf('%s:\r\nIncrease number of examples', ME.message),sprintf('Error: %s', ME.identifier), 'modal');
%            end
            
            if is_error == 0
                if pos_class == 1
                    [X,Y,T,AUC] = perfcurve(dataPR2.groupsAllReal, dataPR2.scores, pos_class);
                else
                    [X,Y,T,AUC] = perfcurve(dataPR2.groupsAllReal, 1./dataPR2.scores, pos_class);
                end
                    
                
                % Находим порог отношения правдоподобий для критерий Пирсона
                e1_pearson = 0.05; % Фиксируем ошибку первого рода (ложное срабатывание)
                [v, i] = sort(abs(X-e1_pearson));
                t_pearson = T(i(1))
                x_pearson = X(i(1));
                y_pearson = Y(i(1));
                
                % Находим порог отношения правдоподобий для критерий Минимакс
                [v, i] = sort(abs(Y+X-1));
                t_minimax = T(i(1))
                x_minimax = X(i(1));
                y_minimax = Y(i(1));
                
                
                % Находим порог отношения правдоподобий для критерия Байеса
                if bayes_type == 1
                    slope_bayes = 1;
                elseif bayes_type == 2
                    slope_bayes = p_w(n_class_2(2))/p_w(n_class_2(1));
                elseif bayes_type == 3
                    i1 = n_class_2(1); i2 = n_class_2(2);
                    slope_bayes = p_w(i2)/p_w(i1)*(C_matr(i2,i1)-C_matr(i2,i2))/(C_matr(i1,i2)-C_matr(i1,i1));
                end
                
                % Поиск на графике ROC точки с заданным углом накллона
                %             delta = 4;
                %             slope = (Y(delta:end) - Y(1:end-delta+1))./(X(delta:end) - X(1:end-delta+1));
                %             [v,i] = sort( abs(slope - slope_bayes));
                %             x1 = X(i(1)+delta/2);
                %             y1 = Y(i(1)+delta/2);
                %             t1 = T(i(1)+delta/2);
                
                [v,i] = sort( abs(T - slope_bayes));
                t_bayes_map = T(i(1));
                x_bayes_map = X(i(1));
                y_bayes_map = Y(i(1));
                
                % Отображение точек, соответствующих критериям Байеса, Пирсона
                % и минимакс
                
                marker_size = 8;
                
                figure;
                subplot(2,1,1);
                plot(X,Y)
                %            xlabel('False positive rate')
                ylabel('True positive rate')
                hold on; grid on;
                plot([0 1], [1 0], 'r');
                
                type1 = class_data{data_type}{n_class_2(1)}.type;
                type2 = class_data{data_type}{n_class_2(2)}.type;
                do_plot_extra_info = classType ~= 1 || ( ~isempty(find(type1==[1 2], 1)) && ~isempty(find(type2==[1 2], 1)));
                do_plot_extra_info = 1
                
                if do_plot_extra_info
                    plot(x_bayes_map,y_bayes_map, 'ok', 'markersize',marker_size);
                    plot(x_pearson,y_pearson, 'og', 'markersize',marker_size);
                    plot(x_minimax,y_minimax, 'or', 'markersize',marker_size);
                    plot([0 x_bayes_map+(1-y_bayes_map)/t_bayes_map],[-t_bayes_map*x_bayes_map+y_bayes_map 1], 'k');
                end
                %plot(X, fnval(spl, X), 'color', 'g', 'linewidth', 1);
                
                subplot(2,1,2);
                grid on; hold on;
                xlabel('False positive rate')
                ylabel('Threshold')
                plot(X,T)
                
                if do_plot_extra_info
                    plot(x_bayes_map,t_bayes_map, 'ok','markersize',marker_size);
                    plot(x_pearson,t_pearson, 'og', 'markersize',marker_size);
                    plot(x_minimax,t_minimax, 'or', 'markersize',marker_size);
                end
                if max(T) > 1e3
                    axis([min(X) max(X) 0 1e3]);
                end
                title([GetCurrentExpParamsStr(h,1) ]);
            end
        case 6
            [dataPR.dataAll, dataPR.groupsAllReal] = GenerateData(data_type, get_N(N_data,data_type), 1:C(data_type));
            [dataPR.dataTrain, dataPR.groupsTrain] = GenerateData(data_type, get_N(nTrain_data,data_type), 1:C(data_type));
        case 7
            if isempty(h_dlg_set_param) || ~ishandle(h_dlg_set_param)
                h_dlg_set_param = setparameters_dialog('Change parameters');
            else
                figure(h_dlg_set_param);
            end
        case 8
            close all
        case length(h_questions)
            break;
    end
    h = menu(h_caption,h_questions );
end
if ~isempty(h_dlg_set_param) && ishandle(h_dlg_set_param)
    close(h_dlg_set_param);
end

close all;
% Генерация данных для основной задачи классификации (нормальное
% распределение)
function [dataAll, groupsAll] = GenerateData(data_type, N_data, n_class_2)
global class_data;

class_data_cur = class_data{data_type};

for c = 1:length(n_class_2)
    c1 = n_class_2(c);
    class_cur = class_data_cur{c1};
    
    if class_cur.type == 1
        data{c} = mvnrnd(class_cur.par1, class_cur.par2, N_data(c));
    elseif class_cur.type == 2
        obj = gmdistribution(class_cur.par1, class_cur.par2, class_cur.par3);
        data{c} = random(obj,N_data(c));
    elseif class_cur.type == 3
        data{c} = uni_geom_rnd(class_cur.par1,class_cur.par2,N_data(c));
    elseif class_cur.type == 4
        data{c} = curve_rnd(class_cur.par1,class_cur.par2, class_cur.par3, N_data(c));
    end
end

dataAll = [];groupsAll = [];

for c = 1:length(n_class_2)
    dataAll = [dataAll; data{c}];
    groupsAll = [groupsAll c*ones(1, N_data(c))];
end
end
%% Классификаторы
% Пакетный классификатор Байеса
function [k, learners, params] = classify_bayes_all(x_all, dataTrain, groupsTrain, learners, params)
global C theta;
global p_w C_matr bayes_type;
global classType data_type;

p_bayes = zeros(C(data_type), size(x_all,1));
[p_mle,~,learners, params] = get_density(x_all, dataTrain, groupsTrain, classType, 1:C(data_type),learners, params);
    
if bayes_type == 1          % ML
    p_bayes = p_mle;
elseif bayes_type == 2      % MAP
    for c = 1:C(data_type)
        p_bayes(c,:) = p_mle(c,:)*p_w(c);
    end
elseif bayes_type == 3      % Min Risk
    for c = 1:C(data_type)
        for c2 = 1:C(data_type)
            p_bayes(c,:) = p_bayes(c,:) + C_matr(c2,c)*p_w(c2)*p_mle(c2,:);
        end
    end
end

if bayes_type ~= 3
    [~,k] = max(p_bayes);
    [scores,~] = max(p_mle);
    % Отклонение 
    k = k.*(scores >= theta);
else
    [scores,~] = max(p_mle);
    [~,k] = min(p_bayes);
    k = k.*(scores >= theta);
end
end
% Пакетный классификатор Байеса (для нормального распределения) с ответом в
% виде отношения правдоподобий
function [scores,learners,params] = classify_bayes_all_soft(x_all, groupsAllReal2, dataTrain, groupsTrain, learners, params)
global n_class_2 classType;

[p_mle,~,learners,params] = get_density(x_all, dataTrain, groupsTrain, classType, n_class_2, learners, params );
delta = 0;
[scores] = (p_mle(1,:)+delta) ./ (p_mle(2,:)+delta);
%scores(p_mle(1,:)==0) = 0;
scores(p_mle(2,:)==0) = max(scores(~isinf(scores)))+1;
%k = k.*(k>= theta);
end
%% Нахождение плотности распределения
function [p_mle, c_levels, learners, params] = get_density(x_all, dataTrain, groupsTrain, classType, classDiap, learners, params)
global kNN_K kNN_MetricType kNN_kw kNN_Pmink;
global SVM_KernelType SVM_SolverType SVM_C SVM_outlier_fr SVM_pol_degree SVM_MLP_params SVM_auto_scale SVM_scale_value;
global NN_arch_type NN_out_type NN_train_alg NN_hidden_neurons NN_hidden_type NN_perf_type NN_max_fail;
global Tree_MaxNumSplits Tree_MinLeafSize Tree_MinParentSize Tree_MergeLeaves;

if classType == 1
    [p_mle, c_levels] = get_density_true(x_all,classDiap);
elseif classType == 2 || classType == 3
    [p_mle, c_levels] = get_density_par_est(x_all, dataTrain, groupsTrain,classDiap, classType);
elseif classType == 4
    [p_mle, c_levels] = get_knn_dens_est(x_all, dataTrain, groupsTrain,classDiap, kNN_K, kNN_MetricType, kNN_kw, kNN_Pmink );
elseif classType == 5
    [p_mle, c_levels, learners.svms] = get_svm_dens_est(x_all, dataTrain, groupsTrain,classDiap, SVM_KernelType, SVM_SolverType, SVM_C, SVM_outlier_fr, SVM_pol_degree, SVM_MLP_params, SVM_auto_scale, SVM_scale_value, learners.svms, params.svm_need_retrain );
    params.svm_need_retrain = 0;
elseif classType == 6
    [p_mle, c_levels, learners.net] = get_nn_dens_est(x_all, dataTrain, groupsTrain,classDiap, learners.net, NN_arch_type, NN_out_type, NN_hidden_type, NN_perf_type, NN_max_fail, NN_train_alg, NN_hidden_neurons, params.net_need_rebuild , params.net_need_retrain);
    params.net_need_rebuild = 0; params.net_need_retrain = 0;
elseif classType == 7
    [p_mle, c_levels, learners.tree] = get_tree_dens_est(x_all, dataTrain, groupsTrain,classDiap, learners.tree, Tree_MaxNumSplits, Tree_MinLeafSize, Tree_MinParentSize, Tree_MergeLeaves, params.tree_need_retrain);
    params.tree_need_retrain = 0;
end
end
% Истинные плотности распределений
function [p_mle, c_levels] = get_density_true(x_all, classDiap)
global C data_type class_data;

class_data_cur = class_data{data_type};

k = 3;
k_mod = 0.3;

c_levels = zeros(1,length(classDiap));
p_mle = zeros(length(classDiap), size(x_all,1));

for i = 1:length(classDiap)
    c = classDiap(i);
    class_cur = class_data_cur{c};
    c_levels(i) = .1;
    if class_cur.type == 1
        p_mle(i,:) = mvnpdf(x_all, class_cur.par1, class_cur.par2);
        c_levels(i) = 1/((2*pi)*det(class_cur.par2)^0.5)*exp(-k^2/2)*k_mod;
    elseif class_cur.type == 2
        obj = gmdistribution(class_cur.par1, class_cur.par2, class_cur.par3);
        p_mle(i,:) = pdf(obj,x_all);
        for nc = 1:length(class_cur.par3)
            c_levels(i) = c_levels(i) + class_cur.par3(nc)/det(class_cur.par2(:,:,nc)^0.5);
        end
        c_levels(i) = c_levels(i) /(2*pi) *exp(-k^2/2) * k_mod;
    elseif class_cur.type == 3
        p_mle(i,:) = uni_geom_pdf(x_all, class_cur.par1, class_cur.par2);
    elseif class_cur.type == 4
        p_mle(i,:) = curve_pdf(x_all, class_cur.par1, class_cur.par2, class_cur.par3);
        c_levels(i) = max(p_mle(i,:));
    end
end
end
% Параметрическая оценка плотностей распределений
function [p_mle, c_levels] = get_density_par_est(x_all, dataTrain, groupsTrain, classDiap, classType)
global isNaiveBayes_data normType_data;
global  parDistrType_data nCompPar_data gmRegularize_data sharedCov_data replicates_data algStart_data covType_data;
global parzenType_data smoothPar_data;
global data_type;

gmRegularize = get_N(gmRegularize_data,data_type);
p_mle = zeros(length(classDiap), size(x_all,1));
c_levels = zeros(1,length(classDiap));
parDistrType = get_N(parDistrType_data,data_type);
nCompPar = get_N(nCompPar_data,data_type);
isNaiveBayes = get_N(isNaiveBayes_data,data_type);
normType = get_N(normType_data,data_type);
parzenType = get_N(parzenType_data,data_type);
smoothPar = get_N(smoothPar_data,data_type);
sharedCov = get_N(sharedCov_data,data_type);
replicates = get_N(replicates_data,data_type);
algStart = get_N(algStart_data,data_type);
covType = get_N(covType_data,data_type);

for i = 1:length(classDiap)
    c = classDiap(i);
    xtrainc = dataTrain(groupsTrain == i,:);
    
    Wnorm = eye(2,2);
    if normType(c) == 1
    elseif normType(c) == 2
        Wnorm = diag(diag(cov(xtrainc)))^(-1/2);
    elseif normType(c) == 3
        Wnorm = cov(xtrainc)^(-1/2);
    end
    xtrainc = xtrainc * Wnorm;
    x_allc = x_all * Wnorm;
    
    if isNaiveBayes(c) == 1
        p_mle(i,:) = ones(1,size(x_allc,1)); c_levels(i) = 1;
        for d = 1:size(x_allc,2)
            if classType == 2
                if parDistrType(c) == 1    % Normal
                    [p_mle_d, c_levels_d ] = get_norm_dens_est(x_allc(:,d), xtrainc(:,d));
                elseif parDistrType(c) == 2   %GMM
                    [p_mle_d, c_levels_d ] = get_gmm_dens_est(x_allc(:,d), xtrainc(:,d), nCompPar(c), gmRegularize(c), sharedCov(c), replicates(c), algStart(c), covType(c));
                end
                p_mle(i,:) = p_mle(i,:) .* p_mle_d; c_levels(i) = c_levels(i)*c_levels_d;
            elseif classType == 3   % KDE
                [p_mle_d, c_levels_d ] = get_kde_dens_est(x_allc(:,d), xtrainc(:,d), parzenType(d,c), smoothPar(d,c));
                p_mle(i,:) = p_mle(i,:) .* p_mle_d; c_levels(i) = c_levels(i)*c_levels_d;
            end
        end
    elseif isNaiveBayes(c) == 2
        if classType == 2 
            if parDistrType(c) == 1    % Normal
                [p_mle(i,:), c_levels(i)] = get_norm_dens_est(x_allc, xtrainc);
            elseif parDistrType(c) == 2 % GMM
                [p_mle(i,:), c_levels(i)] = get_gmm_dens_est(x_allc, xtrainc, nCompPar(c), gmRegularize(c), sharedCov(c), replicates(c), algStart(c), covType(c));
            end
        elseif classType == 3   % KDE
            [p_mle(i,:), c_levels(i)] = get_kde_dens_est(x_allc, xtrainc, parzenType(:,c), smoothPar(:,c));
        end
    end
end
end
% Оценка нормальной плотности
function [p_mle,c_levels] = get_norm_dens_est(x, xTrain)
k = 3;
k_mod = 0.3;

if size(x,2) == 1
    mu = mean(xTrain);
    sigma = cov(xTrain);
    p_mle = mvnpdf(x, mu, sigma)';
    c_levels = sqrt(k_mod/(2*pi*sigma)*exp(-k^2/2));
else
    mu = mean(xTrain);
    sigma = cov(xTrain);
    p_mle = mvnpdf(x, mu, sigma)';
    c_levels = 1/((2*pi)*det(sigma)^0.5)*exp(-k^2/2)*k_mod;
end
end
% Оценка плотности GMM
function [p_mle,c_levels] = get_gmm_dens_est(x, xTrain, nComp, gmRegularize, sharedCov, replicates, algStart, covType)

k = 3;
k_mod = 0.3;
algsStart = {'randSample', 'plus' };
covTypes = {'full', 'diagonal' };

if size(x,2) == 1
    obj = gmdistribution.fit(xTrain, nComp, 'Start', algsStart{algStart}, 'Replicates', replicates, 'CovType', covTypes{covType}, 'SharedCov', logical(sharedCov), 'Regularize', gmRegularize );
    p_mle = pdf(obj, x)';
    c_levels = 0;
    for nc = 1:nComp
        ncc = nc;
        if logical(sharedCov)
            ncc = 1;
        end
        if covType == 1
            denumVal = sqrt(obj.Sigma(:,:,ncc));
        else
            denumVal = (prod(obj.Sigma(:,:,ncc))^0.5);
        end
        c_levels = c_levels + obj.ComponentProportion(nc)/denumVal;
    end
    c_levels = c_levels * sqrt(k_mod/(2*pi)*exp(-k^2/2));
else
    obj = gmdistribution.fit(xTrain, nComp, 'Start', algsStart{algStart}, 'Replicates', replicates, 'CovType', covTypes{covType}, 'SharedCov', logical(sharedCov), 'Regularize', gmRegularize );
    p_mle = pdf(obj,x);
    c_levels = 0;
    for nc = 1:nComp
        ncc = nc;
        if logical(sharedCov)
            ncc = 1;
        end
        if covType == 1
            denumVal = (det(obj.Sigma(:,:,ncc))^0.5);
        else
            denumVal = (prod(obj.Sigma(:,:,ncc))^0.5);
        end
        
        c_levels = c_levels + obj.ComponentProportion(nc)/denumVal;
    end
    c_levels = c_levels /(2*pi) * exp(-k^2/2) * k_mod;
end
    
end
% Оценка плотности KDE
function [p_mle, c_levels] = get_kde_dens_est(x, xTrain, parzenType, smoothPar)
parzen_funcs = {@parzen_window_rect, @parzen_window_gauss, @parzen_window_tri, @parzen_window_exp, @parzen_window_koshi, @parzen_window_regenerate};
N = size(x,1);
Ntrain = size(xTrain,1);
V = prod(smoothPar);
if size(x,2) == 1
    % 1-мерная плотность
    distAr = parzen_funcs{parzenType}(bsxfun(@minus, x, xTrain')/smoothPar); % nTest*nTrain
    p_mle = sum(distAr') / (Ntrain*V);
else
    % многомерная плотность
    d = size(x,2);
    for i = 1:d
        if i == 1
            distAr = parzen_funcs{parzenType(i)}(bsxfun(@minus, x(:,i), xTrain(:,i)')/smoothPar(i)); % nTest*nTrain
        else
            distAr = distAr .* parzen_funcs{parzenType(i)}(bsxfun(@minus, x(:,i), xTrain(:,i)')/smoothPar(i)); % nTest*nTrain
        end
    end
    p_mle = sum(distAr') / (Ntrain*V);
end
c_levels = min(p_mle(p_mle~=0));
if isempty(c_levels)
    c_levels = 0;
end
end
% Оценка плотности kNN
function [p_mle, c_levels] = get_knn_dens_est(x_all, dataTrain, groupsTrain, classDiap, kNN_K, kNN_MetricType, kNN_kw, kNN_Pmink )
distances = {'euclidean', 'seuclidean', 'cityblock', 'chebychev', 'minkowski', 'mahalanobis', 'cosine', 'correlation', 'spearman', 'hamming', 'jaccard'};

N = size(x_all,1);
p_mle = zeros(length(classDiap), N);
c_levels = zeros(1,length(classDiap));

if kNN_MetricType == 2
    [IDX,D] = knnsearch(dataTrain, x_all, 'K', kNN_K, 'Distance', distances{kNN_MetricType}, 'Scale', [1 kNN_kw]);
elseif kNN_MetricType == 5
    [IDX,D] = knnsearch(dataTrain, x_all, 'K', kNN_K, 'Distance', distances{kNN_MetricType}, 'P', kNN_Pmink);
else
    [IDX,D] = knnsearch(dataTrain, x_all, 'K', kNN_K, 'Distance', distances{kNN_MetricType});
end
rd = D(:,kNN_K)';
V = pi * rd .^ 2;
for i = 1:length(classDiap)
    if kNN_K == 1
        p_mle(i,:) = (groupsTrain(IDX)==i) ./ (N*V);
    else
        p_mle(i,:) = sum(groupsTrain(IDX)'==i) ./ (N*V);
    end
    c_levelsi = min(p_mle(i,(p_mle(i,:)~=0)));
    if isempty(c_levelsi)
        c_levelsi = 0;
    end
    c_levels(i) = c_levelsi;
end
end
% Оценка псевдо-плотности SVM
function [p_mle, c_levels, SVMModels] = get_svm_dens_est(x_all, dataTrain, groupsTrain,classDiap, SVM_KernelType, SVM_SolverType, SVM_C, SVM_outlier_fr, SVM_pol_degree, SVM_MLP_params, SVM_auto_scale, SVM_scale_value, SVMModels, svm_need_retrain )
kernels = {'linear', 'polynomial', 'rbf', 'mlp_kernel_mine' };
solvers = {'SMO', 'ISDA', 'L1QP' };

N = size(x_all,1);
p_mle = zeros(length(classDiap), N);
c_levels = zeros(1,length(classDiap));
if svm_need_retrain
    SVMModels = cell(length(classDiap),1);
end

if SVM_auto_scale == 1
    kernel_scale = 'auto';
else
    kernel_scale = SVM_scale_value;
end

for i = 1:length(classDiap)
    if svm_need_retrain
        if SVM_KernelType == 2
            SVMModels{i} = fitcsvm(dataTrain,groupsTrain==i,'ClassNames',[false true],'KernelFunction',kernels{SVM_KernelType}, 'KernelScale', kernel_scale, 'BoxConstraint',SVM_C, 'Solver', solvers{SVM_SolverType}, 'OutlierFraction',SVM_outlier_fr, 'PolynomialOrder',SVM_pol_degree );
        else
            SVMModels{i} = fitcsvm(dataTrain,groupsTrain==i,'ClassNames',[false true],'KernelFunction',kernels{SVM_KernelType}, 'KernelScale', kernel_scale, 'BoxConstraint',SVM_C, 'Solver', solvers{SVM_SolverType}, 'OutlierFraction',SVM_outlier_fr);
        end
    end
    [~,score] = predict(SVMModels{i}, x_all);
    p_mle(i,:) = score(:,2)';
end
% Приведение псевдо-плотности к диапазону [0 1]
minp = min(min(p_mle));
maxp = max(max(p_mle));
p_mle = (p_mle - minp) / (maxp-minp); 

for i = 1:length(classDiap)
    c_levelsi = min(p_mle(i,(p_mle(i,:)~=0)));
    if isempty(c_levelsi)
        c_levelsi = 0;
    end
    c_levels(i) = c_levelsi;
end
end
function    [p_mle, c_levels, net] = get_nn_dens_est(x_all, dataTrain, groupsTrain, classDiap, net, NN_arch_type, NN_out_type, NN_hidden_type, NN_perf_type, NN_max_fail, NN_train_alg, NN_hidden_neurons, net_need_rebuild , net_need_retrain)
if net_need_rebuild
    net = create_neural_network(NN_arch_type, NN_out_type,NN_hidden_type, NN_hidden_neurons);
end
if net_need_retrain
    net = train_neural_network(net, dataTrain, groupsTrain, NN_out_type, NN_perf_type, NN_max_fail, NN_train_alg);
end

p_mle = sim_neural_network(net, x_all);

for i = 1:length(classDiap)
    c_levelsi = min(p_mle(i,(p_mle(i,:)~=0)));
    if isempty(c_levelsi)
        c_levelsi = 0;
    end
    c_levels(i) = c_levelsi;
end
end
function [p_mle, c_levels, tree] = get_tree_dens_est(x_all, dataTrain, groupsTrain, classDiap, tree, Tree_MaxNumSplits, Tree_MinLeafSize, Tree_MinParentSize, Tree_MergeLeaves, tree_need_retrain)
if tree_need_retrain
    tree = train_decision_tree(dataTrain, groupsTrain, Tree_MaxNumSplits, Tree_MinLeafSize, Tree_MinParentSize, Tree_MergeLeaves);
end
p_mle = sim_desicion_tree(tree, x_all);

for i = 1:length(classDiap)
    c_levelsi = min(p_mle(i,(p_mle(i,:)~=0)));
    if isempty(c_levelsi)
        c_levelsi = 0;
    end
    c_levels(i) = c_levelsi;
end
end
%% KDE - вспомогательные функции
function y = parzen_window_rect(u)
y = abs(u) <= 0.5;
end
function y = parzen_window_gauss(u)
y = 1/sqrt(2*pi)*exp(-u.^2/2);
end
function y = parzen_window_tri(u)
y = (1-abs(u)).*(abs(u)<1);
end
function y = parzen_window_exp(u)
y = 0.5*exp(-abs(u));
end
function y = parzen_window_koshi(u)
y = 1/pi./(1+u.*u);
end
function y = parzen_window_regenerate(u)
y = 1/(2*pi)*(sin(u/2)./(u/2)).^2;
end
function hi = CalcSmoothPars(xtrain, typeAlg, typeFeatures, parzenType, smoothPar, h_diap)
N = size(xtrain,1);
d = size(xtrain,2);
h = zeros(1,d);

if typeAlg == 1
    % Par1
    for i = 1:d
        h(i) = 1.06 * std(xtrain(:,i)) * N^(-1/5);
    end
elseif typeAlg == 2
    % Par2
    for i = 1:d
        A = min(std(xtrain(:,i)), iqr(xtrain(:,i))/1.34);
        h(i) = 0.9 * A * N^(-1/5);
    end
elseif typeAlg == 3
    % Likelihood Crossvalidation
    f_optimize = @(h)opt_mlcv(xtrain, h, parzenType);
    if typeFeatures == 1
        [H1,H2] = meshgrid(h_diap{1},h_diap{2});
        hh = [H1(:) H2(:)];
    elseif typeFeatures == 2
        hh = [h_diap smoothPar(2)*ones(size(h_diap))];
    elseif typeFeatures == 3
        hh = [smoothPar(1)*ones(size(h_diap)) h_diap];
    end
    fh = f_optimize(hh);
    [~,imax] = max(fh);
    h = hh(imax,:);
end
if typeFeatures == 1
    hi = h;
elseif typeFeatures == 2
    hi = h(1);
elseif typeFeatures == 3
    hi = h(2);
end
end
function y = opt_mlcv(xtrain, h, parzenType)
parzen_funcs = {@parzen_window_rect, @parzen_window_gauss, @parzen_window_tri, @parzen_window_exp, @parzen_window_koshi, @parzen_window_regenerate};
N = size(xtrain,1);
Nh = size(h,1);
d = size(xtrain,2);

for i = 1:d
    distAr{i} = bsxfun(@minus, xtrain(:,i), xtrain(:,i)'); % nTest*nTrain
end

y = zeros(Nh,1);

for k = 1:Nh
    hi = h(k,:);
    V = prod(hi);
    for i = 1:d
        if i == 1
            pn = parzen_funcs{parzenType(i)}(distAr{i}/hi(i));
        else
            pn = pn .* parzen_funcs{parzenType(i)}(distAr{i}/hi(i));
        end
    end
    pn(eye(N,N)==1) = 0;
    p_mle = sum(pn') / (N*V);
    % Для прямоугольных окон заменим нулевые вероятности (при их количестве не более 5%) на некоторые
    % маленькие конечные значения в целях вычислительной устойчивости
    i_p_0 = find(p_mle==0);
    p_krit = 0.05;
    
    if length(i_p_0) < N*p_krit && ~isempty(i_p_0) && (parzenType(1)==1 || parzenType(2)==1)
        eps1 = 1e-2;
        p_mle(p_mle==0)=eps1/(N*V);
    end
    % y = prod(p_mle)
    y(k) = sum(log(p_mle));
end
end
%% SVM - прочее
%% NN - функции
function net = create_neural_network(NN_arch_type, NN_out_type, NN_hidden_type, NN_hidden_neurons)
net = [];
if NN_arch_type == 1
    % ff
     net = feedforwardnet(NN_hidden_neurons);
elseif NN_arch_type == 2
     net = cascadeforwardnet(NN_hidden_neurons);
    %cascade
end

outTypes = {'tansig', 'softmax', 'satlin', 'logsig', 'purelin'};
hiddenTypes = {'tansig', 'logsig', 'poslin', 'satlin', 'satlins', 'purelin'};

net.layers{length(net.layers)}.transferFcn = outTypes{NN_out_type}

for i = 1:length(net.layers)-1
    net.layers{i}.transferFcn = hiddenTypes{NN_hidden_type}
end


end
function net = train_neural_network(net, dataTrain, groupsTrain, NN_out_type, NN_perf_type, NN_max_fail, NN_train_alg)

net = SetTrainParam(net, NN_train_alg, NN_perf_type, NN_max_fail );

C = max(groupsTrain);
NS_out = zeros(C,length(groupsTrain));

if NN_out_type == 1 || NN_out_type == 2
    for i = 1:C
        NS_out(i,:) = (groupsTrain == i);
    end
    net = train(net, dataTrain', NS_out);
end
end
function y = sim_neural_network(net, dataAll)
    y = sim(net, dataAll');
end
function net = SetTrainParam(net, type_train_func, type_perform_fcn, NN_max_fail)
trainf_fcns = {'traingd','traingda', 'traingdm', 'traingdx', 'trainrp'... % 1-5
    'traincgf', 'traincgb', 'traincgp', 'trainscg',... % 6-9
    'trainlm', 'trainbfg', 'trainoss', 'trainbr' };   % 10-13
perform_fcns = {'mae', 'mse', 'sae', 'sse', 'crossentropy'};
net.trainfcn = trainf_fcns{type_train_func};        % Функция обучения
net.performfcn = perform_fcns{type_perform_fcn};    % Функция вычисления ошибки

net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotconfusion','plotroc'};

net.outputs{length(net.layers)}.processFcns = {'removeconstantrows','mapminmax'};

if type_perform_fcn == 5
    net.performParam.regularization = 0.1;
    net.performParam.normalization = 'none';
    net.outputs{length(net.layers)}.processParams{2}.ymin = 0;
end;

trainParam = net.trainParam;

trainParam.epochs = 1000;           % Максимальное значение числа эпох обучения
trainParam.time = Inf;              % Максимальное время обучения
trainParam.goal = 0;                % Целевое значение ошибки
trainParam.min_grad = 1e-05;        % Значение градиента для останова
trainParam.max_fail = NN_max_fail;            % Максимальное число эпох для раннего останова

% Параметры визуализации обучения
trainParam.showWindow = true;       % Показывать окно или нет
trainParam.showCommandLine = false; % Выводить в командную строку или нет
trainParam.show = 25;               % Частота обновления - через сколько эпох

switch type_train_func
    case 1
        % traingd - Градиентный спуск
        trainParam.lr = 0.2;               % !Скорость обучения
    case 2
        % traingda - Градиентный спуск c адаптацией
        trainParam.lr = 0.01;               % !Скорость обучения (изначальная)
        trainParam.lr_inc = 1.05;           % !Коэффициент увеличения скорости обучения
        trainParam.lr_dec = 0.7;            % !Коэффициент уменьшения скорости обучения
        trainParam.max_perf_inc  = 1.04;    % !Допустимый коэффициент изменения ошибки 
                                                % при его превышении скорость уменьшается в lr_dec раз, коэффициенты не изменяются, в противном случае коэффициенты изменяются
                                                % Если текущая ошибка меньше предыдущей, то скорость увеличивается в lr_inc раз
    case 3
        % traingm - Градиентный спуск c адаптацией
        trainParam.lr = 0.01;               % !Скорость обучения
        trainParam.mc = 0.9;                % !Момент инерции (от 0 до 1), чем он больше тем более плавное изменение коэффициентов
                                                % При mc=0 traingdm переходит в traingd
    case 4
        % traingx - Градиентный спуск c адаптацией и моментом
        trainParam.lr = 0.1;               % !Скорость обучения (изначальная)
        trainParam.mc = 0.6;                % !Момент инерции
        trainParam.lr_inc = 1.05;           % !Коэффициент увеличения скорости обучения
        trainParam.lr_dec = 0.7;            % !Коэффициент уменьшения скорости обучения
        trainParam.max_perf_inc  = 1.04;    % !Допустимый коэффициент изменения ошибки 
    case 5
        % trainrp
        trainParam.lr = 0.01;               % !Скорость обучения (изначальная)
        % Параметры алгоритма (для поиска значения delta)
        trainParam.delt_inc = 1.2;          % Increment to weight change
        trainParam.delt_dec = 0.5;          % Decrement to weight change
        trainParam.delta0 = 0.07;           % Initial weight change
        trainParam.deltamax = 50.0;         % Maximum weight change
    case {6,7,8, 11, 12}
        % traincgf, traincgp, traincgb, trainbfg, trainoss
        if ~isempty(find(type_train_func == [6 7 8], 1))
            trainParam.searchFcn = 'srchcha';   % !Функция одномерного линейного поиска (srchbac, srchbre, srchgol, srchhyb)
        else
            trainParam.searchFcn = 'srchbac';   % !Функция одномерного линейного поиска (srchbac, srchbre, srchgol, srchhyb)
        end
        % Параметры функции одномерного поиска
        trainParam.scale_tol = 20;         % Divide into delta to determine tolerance for linear search.
        trainParam.alpha = 0.001;           % Scale factor that determines sufficient reduction in perf
        trainParam.beta = 0.1;              % Scale factor that determines sufficiently large step size
        trainParam.delta = 0.01;            % Initial step size in interval location step
        trainParam.gama = 0.1;              % Parameter to avoid small reductions in performance, usually set to 0.1 (see srch_cha)
        trainParam.low_lim = 0.1;           % Lower limit on change in step size
        trainParam.up_lim = 0.5             % Upper limit on change in step size
        trainParam.max_step = 100;           % Maximum step length
        trainParam.min_step = 1.0e-6;        % Minimum step length
        trainParam.bmax = 26;               % Maximum step size
        if type_train_func == 11
            trainParam.batch_frag = 0;          % In case of multiple batches, they are considered independent. Any nonzero value implies a fragmented batch, so the final layer's conditions of a previous trained epoch are used as initial conditions for the next epoch.
        end;
    case 9
        % trainscgf
        trainParam.sigma = 5e-5;            % Изменение весов для аппроксимации второй производной
        trainParam.lambda = 5e-7;           % Параметр для регуляризации при плохой обусловенности матрицы Гессе
    %-------------------------------------------------------%
    %-------Методы переменной метрики-----------------------%
    %-------------------------------------------------------%
    case {10, 13}
        % trainlm, trainbr
        % Параметры алгоритма (для поиска значения mu)
        trainParam.mu = 0.001;              % Initial mu
        trainParam.mu_dec = 0.1;            % mu decrease factor
        trainParam.mu_inc = 10;             % mu increase factor
        trainParam.mu_max = 1e10;           % Maximum mu
end;
net.trainParam = trainParam;
end
%% Tree - функции
function tree = train_decision_tree(dataTrain, groupsTrain, Tree_MaxNumSplits, Tree_MinLeafSize, Tree_MinParentSize, Tree_MergeLeaves)
on_offs = {'on', 'off'};
tree = fitctree(dataTrain, groupsTrain, 'MaxNumSplits', Tree_MaxNumSplits, 'MinLeafSize', Tree_MinLeafSize, 'MinParentSize', Tree_MinParentSize, 'MergeLeaves', on_offs{2-Tree_MergeLeaves});
end
function y = sim_desicion_tree(tree, dataAll )
[label,score,node,cnum] = predict(tree, dataAll);
y = score';
end
%% Генерация равномерных распределений внутри геометрических фигур
function y = uni_geom_rnd(uni_data_add, uni_data_remove, N)
N_int = 1e4;
[rect, f_val, sq] = find_uni_pdf(uni_data_add, uni_data_remove, N_int);
if N < 200
    k_Ng = 5;
else
    k_Ng = 3;
end
Ng = ceil(N*f_val*sq*k_Ng);

x1min = rect(1); x1max = rect(2); x2min = rect(3); x2max = rect(4);
y = ones(Ng,1)*[x1min x2min] + rand(Ng,2)*[x1max-x1min 0; 0 x2max-x2min];

x = zeros(Ng,1);
for i = 1:length(uni_data_add)
    x = x | check_geom(y, uni_data_add{i});
end
for i = 1:length(uni_data_remove)
    x = x & ~check_geom(y, uni_data_remove{i});
end

y = y(x~=0,:);
y = y(1:N,:);
end
% Плотность равмноерного распределения внутри геометрической области
function y = uni_geom_pdf(x, uni_data_add, uni_data_remove)
% Находим вначале прямоугольник с областью, включающей в себя все
% под-области
N_int = 1e4;
[rect, f_val, sq] = find_uni_pdf(uni_data_add, uni_data_remove, N_int);
y = zeros(length(x),1);

for i = 1:length(uni_data_add)
    y = y | check_geom(x, uni_data_add{i});
end
for i = 1:length(uni_data_remove)
    y = y & ~check_geom(x, uni_data_remove{i});
end
y = y * f_val;
end
% Плотность распределения вокруг кривой (принимается равномерной)
function y = curve_rnd(curve_param, distr_param, p_param, N)
global Np_calc_length;
N_curves = length(distr_param);
% Определение числа точек для каждой кривой
if p_param{1} == 1
    % {1} - число точек пропорционально длине кривой
    % Вычисление длин кривых
    for i = 1:N_curves
        [xx,yy] = curve_get_points(curve_param{i}, Np_calc_length);
        len_ci(i) = sum(abs(diff(complex(xx,yy))));
    end
    Ni = len_ci/sum(len_ci)*N;
elseif p_param == 2
    % {2,  [p1 p2 ... pN]} - число точек для каждой кривой задается через pi
    Ni = p_param{2}*N;
    Ni(end) = N - sum(Ni(1:end-1));
end

for i = 1:length(Ni)
    Ni(i) = int32(Ni(i));
end

yy = cell(length(curve_param),1);
for i = 1:N_curves
   [x,y] =  curve_get_points(curve_param{i}, Ni(i));
   yy{i} = [x' y'];
end

for i = 1:N_curves
    distr_part_par = distr_param{i};
    if distr_part_par{1} == 1 
        s1 = distr_part_par{2}(1); s2 = distr_part_par{2}(2); r = distr_part_par{2}(3);
        % {1, [s1 s2 r]} - нормальное распределение с СКО s1, s2 и к-том корреляции r
        dy = mvnrnd([0 0], [s1^2 r*s1*s2; r*s1*s2 s2^2], Ni(i));
    elseif distr_part_par{1} == 2
        % {2, [r1 r2]} - равмномерное распределение в прямоугольнике, отстоящем на r1 вправо-влево, r2 - вверх-вниз
        r1 = distr_part_par{2}(1); r2 = distr_part_par{2}(2);
        dy = ones(Ni(i),1)*[-r1 -r2] + rand(Ni(i),2)*[2*r1 0; 0 2*r2];
    elseif distr_part_par{1} == 3
        % {3, r} - равмномерное распределение в круге радиуса r
        r = distr_part_par{2}(1);
        r_i = rand(Ni(i),1)*r;
        phi_i = rand(Ni(i),1)*2*pi;
        dy = [r_i.*cos(phi_i) r_i.*sin(phi_i)];
        
    end
    yy{i} = yy{i} + dy;
end

y = zeros(N,2); ct = 1;
for i = 1:N_curves
    y(ct:ct+Ni(i)-1,:) = yy{i}; ct = ct + Ni(i);
end
end
function [x,y] = curve_get_points(curve_part_par, N)

if curve_part_par{1} == 1
    % {1, tstart, tfinish, @fx(t), @fy(t)} - описание в параметрической форме x=fx(t), y=fy(t), tstart < t < tfinish
    t_start = curve_part_par{2};
    t_finish = curve_part_par{3};
    tt = linspace(t_start, t_finish, N);
    fx = curve_part_par{4};
    fy = curve_part_par{5};
    x = fx(tt);
    y = fy(tt);
elseif curve_part_par{1} == 2
    xx = curve_part_par{2};
    yy = curve_part_par{3};
    
    if length(xx) == 1
        x = xx*ones(N,1);
        y = yy*ones(N,1);
    else
        % Интерполяция сплайнами
        tt = linspace(0,1, length(xx));
        pp = spline(tt,[xx;yy]);
        tt2 = linspace(0,1, N);
        zz = ppval(pp, tt2);
        x = zz(1,:);
        y = zz(2,:);
%        plot(x,y); hold on; plot(xx,yy, 'or');
        
        % {2, xv, yv} - описание в форме последовательности координат точек {xv{i},yv{i}}
    end
end
end

function y = curve_pdf(x, curve_param, distr_param, p_param)
% Находим вначале прямоугольник с областью, включающей в себя все
% под-области
N_int = 1e4;
N_curves = length(distr_param);
[~, f_val, ~] = find_curve_pdf(curve_param, distr_param, p_param, N_int);

y = zeros(length(x),1);

for i = 1:N_curves
    if length(p_param{:}) == 1
        pp = 1/N_curves;
    else
        pp = p_param{i};
    end

    y = y | check_curve(x, curve_param{i}, distr_param{i}, pp);
end

y = y * f_val;
end
% Вспомогательная функция для приближенного определения плотности
% равномерного распределения по площади
function [rect, f_val, sq] = find_uni_pdf(uni_data_add, uni_data_remove, N_int)
x1min = inf; x2min = inf; x1max = -inf; x2max = -inf;
for i = 1:length(uni_data_add)
    uni_data_i = uni_data_add{i};
    [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect(uni_data_i);
    x1min = min([x1min x1min_g]);
    x1max = max([x1max x1max_g]);
    x2min = min([x2min x2min_g]);
    x2max = max([x2max x2max_g]);
end
x_int = ones(N_int,1)*[x1min x2min] + rand(N_int,2)*[x1max-x1min 0; 0 x2max-x2min];
y_int = zeros(length(x_int),1);

for i = 1:length(uni_data_add)
    y_int = y_int | check_geom(x_int, uni_data_add{i});
end
for i = 1:length(uni_data_remove)
    y_int = y_int & ~check_geom(x_int, uni_data_remove{i});
end

rect = [x1min x1max x2min x2max];
sq = (x1max-x1min)*(x2max-x2min);
sq1 = sq*sum(y_int)/length(x_int);
f_val = 1/sq1;
end
function [rect, f_val, sq] = find_curve_pdf(curve_param, distr_param, p_param, N_int)
global axis_rect data_type;
N_curves = length(distr_param);

rect = axis_rect{data_type};
x1min = rect(1);
x1max = rect(2);
x2min = rect(3);
x2max = rect(4);

x_int = ones(N_int,1)*[x1min x2min] + rand(N_int,2)*[x1max-x1min 0; 0 x2max-x2min];
y_int = zeros(length(x_int),1);


for i = 1:N_curves
    if length(p_param{:}) == 1
        pp = 1/N_curves;
    else
        pp = p_param{i};
    end

    y_int = y_int | check_curve(x_int, curve_param{i}, distr_param{i}, pp);
end

sq = (x1max-x1min)*(x2max-x2min);
sq1 = sq*sum(y_int)/length(x_int);
f_val = 1/sq1;
end
% Функции определения попадания в геометрическую фигуру, заданную
% координатами и параметрами
function y = check_geom(x_int, uni_data_i)
if uni_data_i{1} == 1
    y = check_rect(x_int, uni_data_i{2}, uni_data_i{3});
elseif uni_data_i{1} == 2
    y = check_circle(x_int, uni_data_i{2}, uni_data_i{3});
elseif uni_data_i{1} == 3
    y = check_ellipse(x_int, uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
elseif uni_data_i{1} == 4
    y = check_triangle(x_int, uni_data_i{2}, uni_data_i{3}, uni_data_i{4}, uni_data_i{5});
elseif uni_data_i{1} == 5
    y = check_polygon(x_int, uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
end
end
function y = check_rect(x, rect_data, alpha)
alpha_rad = pi/180*alpha;
center = [(rect_data(1)+rect_data(2))/2 (rect_data(3)+rect_data(4))/2];
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
x2 = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = x2(:,1) >= rect_data(1) & x2(:,1) <= rect_data(2) & x2(:,2) >= rect_data(3) & x2(:,2) <= rect_data(4);
end
function y = check_circle(x, center, r)
y = (x(:,1)-center(1)).^2 + (x(:,2)-center(2)).^2 <= r^2;
end
function y = check_ellipse(x, center, r, alpha)
alpha_rad = pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
x = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = (x(:,1)-center(1)).^2/r(1)^2 + (x(:,2)-center(2)).^2/r(2)^2 <= 1;
end
function y = check_triangle(x, p1, p2, p3, alpha)
% если сумма расстояний от каждой вершины до точки меньше периметра, то внутри, иначе - вовне
%per = sum((p1 - p2).^2,2).^0.5 + sum((p1 - p3).^2,2).^0.5 + sum((p2 - p3).^2,2).^0.5;
%ind = sum((x - ones(length(x),1)*p1).^2, 2).^0.5 + sum((x - ones(length(x),1)*p2).^2,2).^0.5 + sum((x - ones(length(x),1)*p3).^2,2).^0.5;
%y = ind < per
alpha_rad = pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*(p1+p2+p3);
x = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = inpolygon(x(:,1), x(:,2), [p1(1) p2(1) p3(1) p1(1)], [p1(2) p2(2) p3(2) p1(2)]);
end
function y = check_polygon(x, xv, yv, alpha)
% если сумма расстояний от каждой вершины до точки меньше периметра, то внутри, иначе - вовне
%per = sum((p1 - p2).^2,2).^0.5 + sum((p1 - p3).^2,2).^0.5 + sum((p2 - p3).^2,2).^0.5;
%ind = sum((x - ones(length(x),1)*p1).^2, 2).^0.5 + sum((x - ones(length(x),1)*p2).^2,2).^0.5 + sum((x - ones(length(x),1)*p3).^2,2).^0.5;
%y = ind < per
alpha_rad = pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*[sum(xv(1:end-1)) sum(yv(1:end-1))];
x = (x - ones(size(x,1),1)*center) * Mrot + ones(size(x,1),1)*center;
y = inpolygon(x(:,1), x(:,2), xv, yv);
end

function y = check_curve(x, curve_param, distr_param, p_param)
global Np_calc_length k_curve deltaT3;
%minD = cell(N_curves,1);
y = zeros(size(x,1),1);
[xx,yy] = curve_get_points(curve_param, Np_calc_length);

deltaNew = deltaT3;

[xy,minD,t_a] = distance2curve([xx' yy'],x);


if distr_param{1} == 1
    %D = pdist2(x, [xx' yy']);
    %minD = min(D,[],2);
    s1 = distr_param{2}(1); s2 = distr_param{2}(2); r = distr_param{2}(3);
    y = y | minD < 3*k_curve*sqrt(s1^2+s2^2);
    y = y | minD <= deltaNew;
elseif distr_param{1} == 2
    eps = 0;
    r1 = distr_param{2}(1); r2 = distr_param{2}(2);
    %Dx = pdist2(x(:,1), xx');
    %fx = Dx+eps <= k_curve*r1;
    %Dy = pdist2(x(:,2), yy');
    %fy = Dy+eps <= k_curve*r2;
    %fxy = fx & fy | fx1 & fy1;
    
    y = y | (minD < sqrt(r1^2 +r2^2));
elseif distr_param{1} == 3
    %D = pdist2(x, [xx' yy']);
    %minD = min(D,[],2);
    r = distr_param{2}(1);
    y = y | minD < k_curve*r;
    %y = y | minD <= deltaNew;
end
end


function y = check_curve_old(x, curve_param, distr_param, p_param)
global Np_calc_length k_curve deltaT3;
%minD = cell(N_curves,1);
y = zeros(size(x,1),1);
[xx,yy] = curve_get_points(curve_param, Np_calc_length);

deltaNew = deltaT3

[xy,distance,t_a] = distance2curve([xx' yy'],x)


if distr_param{1} == 1
    D = pdist2(x, [xx' yy']);
    minD = min(D,[],2);
    s1 = distr_param{2}(1); s2 = distr_param{2}(2); r = distr_param{2}(3);
    y = y | minD < 3*k_curve*sqrt(s1^2+s2^2);
    y = y | minD <= deltaNew;
elseif distr_param{1} == 2
    eps = 0;
    r1 = distr_param{2}(1); r2 = distr_param{2}(2);
    Dx = pdist2(x(:,1), xx');
    fx = Dx+eps <= k_curve*r1;
    fx1 = Dx+eps <= deltaT3/2;
    Dy = pdist2(x(:,2), yy');
    fy = Dy+eps <= k_curve*r2;
    fy1 = Dy+eps <= deltaNew;
    fxy = fx & fy | fx1 & fy1;
    
    y = y | sum(fxy')' > 0;
elseif distr_param{1} == 3
    D = pdist2(x, [xx' yy']);
    minD = min(D,[],2);
    r = distr_param{2}(1);
    y = y | minD < k_curve*r;
    y = y | minD <= deltaNew;
end
end

% Функции определения прямоугольника, включающего в себя заданную
% геометрическую фигуру
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect(uni_data_i)
    if uni_data_i{1} == 1
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_rect(uni_data_i{2}, uni_data_i{3});
    elseif uni_data_i{1} == 2
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_circle(uni_data_i{2}, uni_data_i{3});
    elseif uni_data_i{1} == 3
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_ellipse(uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
    elseif uni_data_i{1} == 4
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_triangle(uni_data_i{2}, uni_data_i{3}, uni_data_i{4}, uni_data_i{5});
    elseif uni_data_i{1} == 5
        [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_polygon(uni_data_i{2}, uni_data_i{3}, uni_data_i{4});
    end
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_rect(rect, alpha)
alpha_rad = -pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = [(rect(1)+rect(2))/2 (rect(3)+rect(4))/2];

p_corn = [rect(1) rect(3); rect(1) rect(4); rect(2) rect(3); rect(2) rect(4)];
p_corn = (p_corn - ones(size(p_corn,1),1)*center) * Mrot + ones(size(p_corn,1),1)*center;
x1min_g = min(p_corn(:,1)); x1max_g = max(p_corn(:,1)); x2min_g = min(p_corn(:,2)); x2max_g = max(p_corn(:,2));
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_circle(center, r)
x1min_g = center(1) - r; x1max_g = center(1) + r; x2min_g = center(2) - r; x2max_g = center(2) + r;
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_ellipse(center, r, alpha)
x1min_g = center(1) - r(1); x1max_g = center(1) + r(1); x2min_g = center(2) - r(2); x2max_g = center(2) + r(2);
[x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_from_rect([x1min_g,x1max_g,x2min_g, x2max_g], alpha);
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_triangle(p1, p2, p3, alpha)
alpha_rad = -pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*(p1+p2+p3);

p_corn = [p1; p2; p3];
p_corn = (p_corn - ones(size(p_corn,1),1)*center) * Mrot + ones(size(p_corn,1),1)*center;
x1min_g = min(p_corn(:,1)); x1max_g = max(p_corn(:,1)); x2min_g = min(p_corn(:,2)); x2max_g = max(p_corn(:,2));
end
function [x1min_g,x1max_g,x2min_g, x2max_g] = get_rect_form_polygon(xv, yv, alpha)
alpha_rad = -pi/180*alpha;
Mrot = [cos(alpha_rad) -sin(alpha_rad); sin(alpha_rad) cos(alpha_rad)];
center = 1/3*[sum(xv(1:end-1)) sum(yv(1:end-1))];

p_corn = [xv' yv'];
p_corn = (p_corn - ones(size(p_corn,1),1)*center) * Mrot + ones(size(p_corn,1),1)*center;
x1min_g = min(p_corn(:,1)); x1max_g = max(p_corn(:,1)); x2min_g = min(p_corn(:,2)); x2max_g = max(p_corn(:,2));
end
% Классификатор Байеса для одного примера и произвольного распределения (не
% используется)
% function k = classify_bayes(x)
% global p_ML C is_null_class theta;
% 
% pp = zeros(C,1);
% 
% for c = 1:C
%     pp(c) = p_ML{c}(x);
% end
% [temp,k] = max(pp);
% if temp < theta
%     k = 0;
%     is_null_class = 1;
% end
%% Подписи к графика, легенда
function str = get_legend(type, type2, type3)
global data_type C is_null_class; 

str = {};

if ~isempty(find(type == [1 2 3 4],1))
    for i = 1:C(data_type)
        str = [str ['Class ' num2str(i)]];
    end
end

if ~isempty(find(type == [3 4],1)) && is_null_class
    str = [str 'Rejected'];
end
end
function str = GetCurrentExpParamsStr(type, type2)
global data_type bayes_type theta p_w C_matr;
global classType;
global data_type_names;

bayes_types_str = {'ML', 'MAP', 'Min Risk'};
main_titles = {'', '', 'T1 Plot. ', 'T2 plot. ', 'ROC. ' };
data_titles = {'Test', 'Test+True Levels', 'Test+Est. Levels', 'Test+True,Est. Levels', 'Train'};

if classType == 1
    pdf_type_title = 'true';
else
    pdf_type_title = 'par';
end

if type == 1
        str = sprintf('Data%s: %s.', data_titles{type2}, data_type_names{data_type} );
elseif type == 2
    str = sprintf('Data: %s. Pdf: %s.', data_type_names{data_type}, pdf_type_title );
elseif type == 3 || type == 4
    if bayes_type == 1
        str = sprintf('Data: %s. Pdf: %s. Cl: %s theta: %.2g', data_type_names{data_type}, pdf_type_title, bayes_types_str{bayes_type}, theta);
    elseif bayes_type == 2
        str = sprintf('Data: %s. Pdf: %s. Cl: %s. Pw: [%.2g %.2g %.2g]. theta: %.2g', data_type_names{data_type}, pdf_type_title, bayes_types_str{bayes_type}, p_w(1), p_w(2), p_w(3), theta);
    elseif bayes_type == 3
        str = sprintf('Data: %s. Pdf: %s.\r\nCl: %s. Pw: [%.2g %.2g %.2g]C: [%.2g %.2g %.2g; %.2g %.2g %.2g; %.2g %.2g %.2g]', data_type_names{data_type}, pdf_type_title, bayes_types_str{bayes_type}, ...
            p_w(1), p_w(2), p_w(3), C_matr(1,1), C_matr(1,2), C_matr(1,3), C_matr(2,1), C_matr(2,2), C_matr(2,3), C_matr(3,1), C_matr(3,2), C_matr(3,3));
    end
elseif type == 5
    str = sprintf('Data: %s, theta: %.2g', data_type_names{data_type}, theta);
end
str = [main_titles{type} str];
end
function nTrain = get_N(N_data, data_type)
global C;
if N_data{1} == 1
    nTrain = N_data{2}*ones(1, C(data_type));
else
    nTrain = N_data{3};
end
end
function data = GetZ1d(hNew, Z, x1, x2)
if hNew == 2
    xd = x1
    xd2 = x2
    Z1d = sum(Z,1)
    x_label = 'x1'
elseif hNew == 3
    xd = x2
    xd2 = x1
    Z1d = sum(Z,2)
    x_label = 'x2'
end
Z1d = Z1d * (xd2(2)-xd2(1))
data = {xd, Z1d, x_label};
end


%% Изменение через меню основных параметров
function d = setparameters_dialog(name, dataTrain, groupsTrain, dataAll, groupsAllReal, net)
global p_w C C_matr bayes_type data_type data_type_names n_class_2 pos_class theta nTrain_data N_data;
global classType parDistrType_data nCompPar_data gmRegularize_data sharedCov_data replicates_data algStart_data covType_data isNaiveBayes_data;
global normType_data parzenType_data smoothPar_data smoothParCalc_data;
global kNN_K kNN_MetricType kNN_TieAlgType kNN_kw kNN_kw_CalcAlg kNN_Pmink;
global SVM_KernelType SVM_SolverType SVM_C SVM_outlier_fr SVM_pol_degree SVM_MLP_params SVM_auto_scale SVM_scale_value;
global NN_arch_type NN_out_type NN_train_alg NN_hidden_neurons NN_hidden_type NN_perf_type NN_max_fail;
global Tree_MaxNumSplits Tree_MinLeafSize Tree_MinParentSize Tree_MergeLeaves;
global dataPR dataPR2;
%% Initialization
N_data_copy = N_data;
Ntrain_data_copy = nTrain_data;
p_w_copy = p_w;
theta_copy = theta;
C_matr_copy = C_matr;
n_class_2_copy = n_class_2;
pos_class_copy = pos_class;
isNaiveBayes_data_copy = isNaiveBayes_data;
parDistrType_data_copy = parDistrType_data;
nCompPar_data_copy = nCompPar_data;
gmRegularize_data_copy = gmRegularize_data;
sharedCov_data_copy = sharedCov_data; 
replicates_data_copy = replicates_data;
algStart_data_copy = algStart_data; 
covType_data_copy = covType_data;
data_type_copy = data_type;
normType_data_copy = normType_data;
parzenType_data_copy = parzenType_data;
smoothPar_data_copy = smoothPar_data;
smoothParCalc_data_copy = smoothParCalc_data;
kNN_K_copy = kNN_K;
kNN_MetricType_copy = kNN_MetricType;
kNN_TieAlgType_copy = kNN_TieAlgType;
kNN_kw_copy = kNN_kw;
kNN_kw_CalcAlg_copy = kNN_kw_CalcAlg;
kNN_Pmink_copy = kNN_Pmink;
SVM_KernelType_copy = SVM_KernelType;
SVM_SolverType_copy = SVM_SolverType;
SVM_C_copy = SVM_C;
SVM_outlier_fr_copy = SVM_outlier_fr;
SVM_pol_degree_copy = SVM_pol_degree;
SVM_MLP_params_copy = SVM_MLP_params;
SVM_auto_scale_copy = SVM_auto_scale;
SVM_scale_value_copy = SVM_scale_value;
NN_arch_type_copy = NN_arch_type;
NN_out_type_copy = NN_out_type;
NN_train_alg_copy = NN_train_alg;
NN_hidden_neurons_copy = NN_hidden_neurons;
NN_hidden_type_copy = NN_hidden_type;
NN_perf_type_copy = NN_perf_type;
NN_max_fail_copy = NN_max_fail;


Tree_MaxNumSplits_copy = Tree_MaxNumSplits;
Tree_MinLeafSize_copy = Tree_MinLeafSize;
Tree_MinParentSize_copy = Tree_MinParentSize;
Tree_MergeLeaves_copy = Tree_MergeLeaves;

dataPR_copy = dataPR;
dataPR2_copy = dataPR2;

h_d = 550; w_d = 420;
d = dialog('Position',[300 50 w_d h_d],'Name',name, 'WindowStyle', 'Normal');
dx_txt = 10; w_txt = 90; h_txt = 25; h_popup = 25; h_button = 25;
dx_edit = dx_txt + w_txt + 10; w_edit = 170; w_txt2 = 40; w_edit2 = 60; w_cb = 20;
dx_txt3 = 20; w_txt4 = w_txt - 20;

ycur = h_d - 30;
deltay = 5;

% Controls
%% N - число примеров
txtN = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_txt],...
    'String','N');
editN = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_edit h_txt],...
    'Callback', @editN_callback);
checkboxNAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value', N_data{1});
popupNClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'String',get_classes_str_arr(1),...
    'Callback', @popups_callback,...
    'Value', 1);
ycur = ycur - h_txt - deltay;
%% nTrain - число обучающих примеров
txtNTrain = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_txt],...
    'String','nTrain');
editNtrain = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_edit h_txt],...
    'Callback', @editNtrain_callback );
checkboxNtrainAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value', nTrain_data{1});
popupNtrainClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'String',get_classes_str_arr(1),...
    'Callback', @popups_callback,...
    'Value', 1);
ycur = ycur - h_txt - deltay;
%% Pw - априорные вероятности
txtPw = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_txt],...
    'String','P(wi)');
editPw = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_edit h_txt],...
    'Callback', @editPw_callback,...
    'String', num2str(p_w'));
ycur = ycur - 2.5*h_txt - deltay;
%% С - матрица стоимости
txtC = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt 2.5*h_txt],...
    'String','C(i,j)');
editC = uicontrol('Parent',d,...
    'Style','edit',...
    'Min', 1, 'Max', 3,...
    'Callback', @editC_callback,...
    'Position',[dx_edit ycur w_edit 2.5*h_txt],...
    'String', num2str(C_matr));
ycur = ycur - h_txt - deltay;
%% Theta - порог срабатывания
txtTheta = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_txt],...
    'String','theta');
editTheta = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_edit h_txt],...
    'Callback', @editTheta_callback,...
    'String', num2str(theta));
ycur = ycur - h_txt - deltay;
%% N2c - номера классов для ROC-кривой
txtN2C = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_txt],...
    'String','2 classes (ROC)');
editN2C = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_edit h_txt],...
    'Callback', @editN2C_callback,...
    'String', num2str(n_class_2));
ycur = ycur - h_txt - deltay;
%% posC - положительный класс для ROC-кривой (1 или 2)
txtposC = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_txt],...
    'String','Pos class (ROC)');
editposC = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_edit h_txt],...
    'Callback', @editposC_callback,...
    'String', num2str(pos_class));
ycur = ycur - h_popup - deltay;
% Тип данных
txtDataType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Data type');
popupDataType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit ycur w_edit h_popup],...
    'Callback', @popupDataType_callback,...
    'String',data_type_names,...
    'Value', data_type);
ycur = ycur - h_popup - deltay;
%% тип Байесового классификатора
txtBayesType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Bayes type');
popupBayesType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit ycur w_edit h_popup],...
    'String',{'ML', 'MAP', 'Bayes'},...
    'Value', bayes_type);
ycur = ycur - h_popup - deltay;
%% Нормализация признаков
txtNormType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Normalization:');
popupNormType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit ycur w_edit h_popup],...
    'Callback', @popupNormType_callback,...
    'String',{'Without', 'Norm variances', 'Whitening'});
checkboxNormTypeAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value', normType_data{1} );
popupNormTypeClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'Callback', @popups_callback,...
    'String',get_classes_str_arr(1),...
    'Value', 1);
ycur = ycur - h_popup - deltay;
%% Наивный или нет классификатор (гипотеза о независимости признаков)
txtIsNaiveBayesType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Naive');
popupIsNaiveBayesType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit ycur w_edit h_popup],...
    'Callback', @popupIsNaiveBayesType_callback,...
    'String',{'Yes', 'No'});
checkboxIsNaiveBayesTypeAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value',isNaiveBayes_data{1} );
popupIsNaiveBayesTypeClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'Callback', @popups_callback,...
    'String',get_classes_str_arr(1),...
    'Value', 1);
ycur = ycur - h_popup - deltay;
%% Тип классификатора
txtClassifierType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Classifier');
popupClassifierType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit ycur w_edit h_popup],...
    'String',get_classifier_types,...
    'Callback', @popups_callback,...
    'Value', classType);
ycur = ycur - h_popup - deltay;
ycur_save = ycur;
%% Распределение для параметрического оценивания
txtposParDistrType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Par distribution');
popupParDistrType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit ycur w_edit h_popup],...
    'String',{'Normal', 'GMM'},...
    'Callback', @popupParDistrType_callback);
checkboxParDistrTypeAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value', parDistrType_data{1});
popupParDistrTypeClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'String',get_classes_str_arr(1),...
    'Callback', @popups_callback,...
    'Value', 1);
ycur = ycur - h_txt - deltay;
%% Число компонент для параметрического оценивания с помощью GMM
txtNCompPar = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_txt],...
    'String','N components');
editNCompPar = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_txt2 h_txt],...
    'Callback', @editNCompPar_callback);
w_txt22 = w_txt2-10;
txtRegPar = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_edit+w_txt2 ycur w_txt22 h_txt],...
    'String','Reg Par');
editRegPar = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit+w_txt2+w_txt22 ycur w_txt2 h_txt],...
    'Callback', @editRegPar_callback);
txtSharedCov = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_edit+2*w_txt2 + w_txt22 ycur w_txt2 h_txt],...
    'String','Share Cov');
checkboxSharedCov = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit+3*w_txt2+1*w_txt22 ycur w_cb h_popup],...
    'Callback', @checkboxSharedCov_callback,...
    'Value', 0);
checkboxNCompParAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value', nCompPar_data{1});
popupNCompParClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'Callback', @popups_callback,...
    'String',get_classes_str_arr(1),...
    'Value', 1);
ycur = ycur - h_txt - deltay;

txtReplicates = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','Replicates');
editReplicates = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_edit ycur w_txt22 h_txt],...
    'Callback', @editReplicates_callback);
txtAlgStart = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_edit+w_txt22 ycur w_txt22 h_popup],...
    'String','Alg1');
popupAlgStart = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit+w_txt22*2 ycur w_edit2 h_popup],...
    'Callback', @popupAlgStart_callback,...
    'String', get_alg_start);
txtCovType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_edit+w_txt22*2+w_edit2 ycur w_txt22 h_popup],...
    'String','Cov Type');
popupCovType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit+w_txt22*3+w_edit2 ycur w_edit2 h_popup],...
    'Callback', @popupCovType_callback,...
    'String', get_cov_types);


%% Типы окон для Парзеновского классификатора
ycur = ycur_save;
txtParzenType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','Parzen windows');
txtParzenX1Type = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4 ycur dx_txt3 h_popup],...
    'String','x1:');
popupParzen1Type = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+dx_txt3+10 ycur w_edit2 h_popup],...
    'Callback', @popupParzen1Type_callback,...
    'String', get_parzen_types);
txtParzenX2Type = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+dx_txt3+w_edit2+10 ycur dx_txt3 h_popup],...
    'String','x2:');
popupParzen2Type = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+w_edit2+2*dx_txt3+20 ycur w_edit2 h_popup],...
    'Callback', @popupParzen2Type_callback,...
    'String', get_parzen_types);
checkboxParzenAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value',parzenType_data{1} );
popupParzenClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'Callback', @popups_callback,...
    'String',get_classes_str_arr(1),...
    'Value', 1);
ycur = ycur - h_txt - deltay;
%% Сглаживающие параметры для парзеновских окон
txtSmoothPar = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','Smooth Par');
txtSmoothParX1 = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4 ycur dx_txt3 h_popup],...
    'String','h1:');
editSmoothParX1 = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt4+dx_txt3+10 ycur w_edit2 h_popup],...
    'Callback', @editSmoothParX1_callback,...
    'String','');
txtSmoothParX2 = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+dx_txt3+w_edit2+10 ycur dx_txt3 h_popup],...
    'String','h2:');
editSmoothParX2 = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt4+w_edit2+2*dx_txt3+20 ycur w_edit2 h_popup],...
    'Callback', @editSmoothParX2_callback,...
    'String', '');
checkboxSmoothParAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value',smoothPar_data{1} );
popupSmoothParClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'Callback', @popups_callback,...
    'String',get_classes_str_arr(1),...
    'Value', 1);
ycur = ycur - h_txt - deltay;
%% Вычисление сглаживающих параметров
dx_txt3 = 20;
w_txt4 = w_txt - 20;
w_edit2 = 50;
txtSmoothParCalc = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','Calc h1, h2');
txtSmoothParCalcAlg = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4 ycur dx_txt3 h_popup],...
    'String','Alg');
popupSmoothParCalcAlg = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+dx_txt3+10 ycur w_edit2 h_popup],...
    'Callback', @popupSmoothParCalcAlg_callback,...
    'String',{'Par1', 'Par2', 'MLE'});
txtSmoothParCalcFeatures = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+dx_txt3+w_edit2+10 ycur dx_txt3 h_popup],...
    'String','hi');
popupSmoothParCalcFeatures = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+2*dx_txt3+w_edit2+10 ycur w_edit2 h_popup],...
    'Callback', @popupSmoothParCalcFeatures_callback,...
    'String',{'Both', 'X1', 'X2'});
w_button2 = 30;
btnSmooParCalcOK = uicontrol('Parent',d,...
    'Position',[dx_txt+w_txt4+2*dx_txt3+2*w_edit2+30 ycur w_button2 h_popup],...
    'String','Calc',...
    'Callback', @btnSmooParCalcOK_callback );
checkboxSmooParCalcAll = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_edit + w_edit + 10 ycur w_cb h_popup],...
    'Callback', @checkboxes_callback,...
    'Value',smoothParCalc_data{1} );
popupSmoothParCalcClass = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_edit + w_edit+20+w_cb ycur w_edit2 h_popup],...
    'Callback', @popups_callback,...
    'String',get_classes_str_arr(2),...
    'Value', 1);
%% k-NN классификатор
ycur = ycur_save;
w_txt2 = 20; w_txt4 = w_txt - 20;
w_txt5 = 35; w_txt6 = 40;
w_popup = 80;
txtKNNPars1 = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','KNN Par-s');
txtKNN_K = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4 ycur w_txt2 h_popup],...
    'String','K:');
editKNN_K = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt4+w_txt2+10 ycur w_edit2 h_popup],...
    'Callback', @editKNN_K_callback,...
    'String',num2str(kNN_K));
txtKNN_Metric = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+w_txt2+w_edit2+15 ycur w_txt5 h_popup],...
    'String','Metric:' );
popupKNN_Metric = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+w_edit2+w_txt2+w_txt5+20 ycur w_popup h_popup],...
    'Callback', @popupKNN_Metric_callback,...
    'String', get_metric_types,...
    'Value', kNN_MetricType);
txtKNN_TieRule = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+w_edit2+w_txt2+w_txt5+w_popup+25 ycur w_txt6 h_popup],...
    'String','Tie rule');
popupKNN_BreakTie = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+w_edit2+w_txt2+w_txt5+w_txt6+w_popup+25 ycur w_popup h_popup],...
    'Callback', @popupKNN_BreakTie_callback,...
    'String', get_break_tie_rules,...
    'Value', kNN_TieAlgType);
ycur = ycur - h_txt - deltay;
%% kNN - параметры метрики
txtKNN_kw = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','k = w2/w1');
editKNN_kw = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt4+w_txt2+10 ycur w_edit2 h_popup],...
    'Callback', @editKNN_kw_callback,...
    'String', num2str(kNN_kw));
txtKNN_PMink = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','p_exp:');
editKNN_PMink = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt4+w_txt2+10 ycur w_edit2 h_popup],...
    'Callback', @editKNN_Pmink_callback,...
    'String',num2str(kNN_Pmink));
w_txt4 = w_txt - 20;
w_edit2 = 60;
txtKNN_WeightsCalcAlg = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+2*w_txt4+w_txt2 ycur w_txt2 h_popup],...
    'String','Alg');
popupKNN_WeightsCalcAlg = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+2*w_txt4+2*w_txt2+10 ycur w_edit2 h_popup],...
    'Callback', @popupKNN_WeightsCalcAlg_callback,...
    'String',{'Filter1', 'Filter2', 'Wrapper'},...
    'Value', kNN_kw_CalcAlg );
w_button2 = 30;
btKNN_WeightsCalcOK = uicontrol('Parent',d,...
    'Position',[dx_txt+2*w_txt4+2*w_txt2+w_edit2+20 ycur w_button2 h_popup],...
    'String','Calc',...
    'Callback', @btKNN_WeightsCalcOK_callback );
%% SVM-классификатор - kernel, solver
ycur = ycur_save;
w_txt4 = w_txt - 30;
w_edit22 = 40;
txtSVMPars1 = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','SVM Par-s');
txtSVM_KernelType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4 ycur w_txt6 h_popup],...
    'String','Kernel:');
popupSVM_KernelType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+w_txt6 ycur w_edit2 h_popup],...
    'Callback', @popupSVM_KernelType_callback,...
    'String', get_SVM_kernel_types,...
    'Value', SVM_KernelType);
txtSVM_SolverType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+w_txt6+w_edit2+5 ycur w_txt6 h_popup],...
    'String','Solver:');
popupSVM_SolverType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt4+2*w_txt6+w_edit2+10 ycur w_edit2 h_popup],...
    'Callback', @popupSVM_SolverType_callback,...
    'String', get_SVM_solver_types,...
    'Value', SVM_SolverType);
txtSVM_C = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+2*w_txt6+w_cb+2.5*w_edit22+25 ycur w_txt6*1.5 h_popup],...
    'String','C:');
editSVM_C = uicontrol('Parent',d,...
    'Style','edit',...
     'Position',[dx_txt+w_txt4+3.5*w_txt6+w_cb+2.5*w_edit22+30 ycur w_edit22 h_popup],...
    'Callback', @editSVM_C_callback,...
    'String',num2str(SVM_C));
ycur = ycur - h_txt - deltay;
%% SVM - параметры ядра
txtSVMPars2 = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt4 h_popup],...
    'String','SVM Kernel');
txtSVM_auto_scale = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4 ycur w_txt6 h_popup],...
    'String','Scale auto:');
checkboxSVM_auto_scale = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position', [dx_txt+w_txt4+w_txt6 ycur w_cb h_popup],...
    'Callback', @checkboxSVM_auto_scale_callback,...
    'Value', SVM_auto_scale );
edit_SVM_scale_value = uicontrol('Parent',d,...
     'Style','edit',...
     'Position',[dx_txt+w_txt4+w_txt6+w_cb+5 ycur w_edit22 h_popup],...
     'Callback', @edit_SVM_scale_value_callback,...
     'String', num2str(SVM_scale_value));
txtSVM_pol_degree = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+w_txt6+w_cb+w_edit22+10 ycur w_txt6 h_popup],...
    'String','Deg:');
edit_SVM_pol_degree = uicontrol('Parent',d,...
     'Style','edit',...
     'Position',[dx_txt+w_txt4+2*w_txt6+w_cb+w_edit22+20 ycur w_edit22 h_popup],...
     'Callback', @edit_SVM_pol_degree_callback,...
     'String', num2str(SVM_pol_degree));
txtSVM_MLP_params = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+w_txt6+w_cb+w_edit22+15 ycur w_txt6 h_popup],...
    'String','MLP:');
edit_SVM_MLP_params = uicontrol('Parent',d,...
     'Style','edit',...
     'Position',[dx_txt+w_txt4+2*w_txt6+w_cb+w_edit22+20 ycur 1.5*w_edit22 h_popup],...
     'Callback', @edit_SVM_MLP_params_callback,...
     'String', num2str(SVM_MLP_params));
txtSVM_outlier_fr = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt4+2*w_txt6+w_cb+2.5*w_edit22+25 ycur w_txt6*1.5 h_popup],...
    'String','Outlier Freq');
edit_SVM_outlier_fr = uicontrol('Parent',d,...
     'Style','edit',...
     'Position',[dx_txt+w_txt4+3.5*w_txt6+w_cb+2.5*w_edit22+30 ycur w_edit22 h_popup],...
     'Callback', @edit_SVM_outlier_fr_callback,...
     'String', num2str(SVM_outlier_fr));
%% Neural networks
ycur = ycur_save;
w_txt4 = w_txt - 30;
w_edit22 = 40; w_edit32 = 60;
w_button2 = 30;
txtNNPars1 = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','NN Architecture');
popupNN_ArchType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt+10 ycur w_edit32 h_popup],...
    'Callback', @popupNN_ArchType_callback,...
    'String', get_NN_arch_types,...
    'Value', NN_arch_type  );
txtNN_OutType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt+w_edit32+15 ycur w_txt6 h_popup],...
    'String','Output:');
popupNN_OutType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt+w_txt6+w_edit32+20 ycur w_edit32 h_popup],...
    'Callback', @popupNN_OutType_callback,...
    'String', get_NN_out_types,...
    'Value', NN_out_type );
txtNN_HiddenType = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt+w_txt6+2*w_edit32+30 ycur w_txt6 h_popup],...
    'String','Hidden:');
popupNN_HiddenType = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt+2*w_txt6+2*w_edit32+40 ycur w_edit32 h_popup],...
    'Callback', @popupNN_HiddenType_callback,...
    'String', get_NN_hidden_types,...
    'Value', NN_hidden_type );
ycur = ycur - h_txt - deltay;
txtNNPars2 = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Hidden neurons');
editNN_hidden_neurons = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt+10 ycur w_edit32 h_popup],...
    'Callback', @editNN_hidden_neurons_callback,...
    'String', num2str(NN_hidden_neurons));
txtNN_train_alg = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt+w_edit32+15 ycur w_txt6 h_popup],...
    'String','Train  Fcn');
popupNN_train_alg = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt+w_txt6+w_edit32+20 ycur w_edit32 h_popup],...
    'Callback', @popupNN_train_alg_callback,...
    'String', get_NN_train_types,...
    'Value', NN_train_alg );
btnNN_train_start = uicontrol('Parent',d,...
    'Position',[dx_txt+w_txt+w_txt6+2*w_edit32+30 ycur w_button2 h_popup],...
    'String','Train',...
    'Callback', @btnNN_train_start_callback );
ycur = ycur - h_txt - deltay;

txtNNValidationChecks = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','Valid. checks');
editNN_ValidationChecks = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt+10 ycur w_edit32 h_popup],...
    'Callback', @editNN_ValidationChecks_callback,...
    'String', num2str(NN_max_fail));
txtNN_perf_alg = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt+w_edit32+15 ycur w_txt6 h_popup],...
    'String','Perf  Fcn');
popupNN_perf_alg = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[dx_txt+w_txt+w_txt6+w_edit32+20 ycur w_edit32 h_popup],...
    'Callback', @popupNN_perf_alg_callback,...
    'String', get_NN_perf_types,...
    'Value', NN_perf_type );
btnNN_view = uicontrol('Parent',d,...
    'Position',[dx_txt+w_txt+w_txt6+2*w_edit32+30 ycur w_button2 h_popup],...
    'String','View',...
    'Callback', @btnNN_view_callback );
btnNN_gensim = uicontrol('Parent',d,...
    'Position',[dx_txt+w_txt+w_txt6+2*w_edit32+w_button2+50 ycur 1.5*w_button2 h_popup],...
    'String','Gensim',...
    'Callback', @btnNN_gensim_callback );
%% Decision Trees
ycur = ycur_save;
w_txt4 = w_txt - 30;
w_edit22 = 40; w_edit32 = 60;
w_button2 = 30; w_button3 = 40;

txtTree_MaxNumSplits = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','MaxNumSplits ');
editTree_MaxNumSplits = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt+10 ycur w_edit32 h_popup],...
    'Callback', @editTree_MaxNumSplits_callback,...
    'String', num2str(Tree_MaxNumSplits));
txtTree_MergeLeaves = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt+w_txt+w_edit32+15 ycur w_txt h_popup],...
    'String','MergeLeaves');
checkboxTree_MergeLeaves = uicontrol('Parent', d,...
    'Style', 'checkbox', ...
    'Position',[dx_txt+w_txt+w_edit32+w_txt+20 ycur w_cb h_popup],...
    'Callback', @checkboxTree_MergeLeaves_callback,...
    'Value', Tree_MergeLeaves );
btnTree_train_start = uicontrol('Parent',d,...
    'Position',[200+w_txt+10 ycur w_button3 h_popup],...
    'String','Train',...
    'Callback', @btnTree_train_start_callback );
ycur = ycur - h_txt - deltay;
txtTree_MinParentSize = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','MinParentSize');
editTree_MinParentSize = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt+10 ycur w_edit32 h_popup],...
    'Callback', @editTree_MinParentSize_callback,...
    'String', num2str(Tree_MinParentSize));
btnTree_view_graph = uicontrol('Parent',d,...
    'Position',[200+w_txt+10 ycur w_button3 h_popup],...
    'String','View',...
    'Callback', @btnTree_view_graph_callback );
ycur = ycur - h_txt - deltay;
txtTree_MinLeafSize = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[dx_txt ycur w_txt h_popup],...
    'String','MinLeafSize');
editTree_MinLeafSize = uicontrol('Parent',d,...
    'Style','edit',...
    'Position',[dx_txt+w_txt+10 ycur w_edit32 h_popup],...
    'Callback', @editTree_MinLeafSize_callback,...
    'String', num2str(Tree_MinLeafSize));
%% Кнопки OK, Отмена
ycur = ycur_save - 2*(h_button + deltay);

ycur = ycur - h_button - 2*deltay;
btnSave = uicontrol('Parent',d,...
    'Position',[20 ycur 50 h_button],...
    'String','Save',...
    'Callback', @btnSave_callback);
btnLoad = uicontrol('Parent',d,...
    'Position',[80 ycur 50 h_button],...
    'String','Load',...
    'Callback', @btnLoad_callback);
btnApply = uicontrol('Parent',d,...
    'Position',[220 ycur 50 h_button],...
    'String','Apply',...
    'Callback', @btnApply_callback);
btnOK = uicontrol('Parent',d,...
    'Position',[280 ycur 50 h_button],...
    'String','OK',...
    'Callback', @ok_callback);
btnCancel = uicontrol('Parent',d,...
    'Position',[340 ycur 50 h_button],...
    'String','Cancel',...
    'Callback', 'delete(gcf)');
%% Открытие диалогового окна
checkboxes_callback;
popups_callback;
% Wait for d to close before running to completion
%uiwait(d);
d;
%% functions basic callbacks
    function editN_callback(edit, events)
        answer = str2num(get(edit,'string')); change = 0;
        if length(answer) ~= 1
        elseif answer < 0
        else
            val = ceil(answer); change = 1;
        end
        
        if change
            if N_data_copy{1} == 1
                N_data_copy{2} = val;
            else
                N_data_copy{3}(get(popupNClass,'Value')) = val;
            end
            set(edit, 'string', num2str(val));
        end
    end
    function editNtrain_callback(edit, events)
        answer = str2num(get(edit,'string')); change = 0;
        if length(answer) ~= 1
        elseif answer < 0
        else
            val = ceil(answer); change = 1;
        end
        
        if change
            if Ntrain_data_copy{1} == 1
                Ntrain_data_copy{2} = val;
            else
                Ntrain_data_copy{3}(get(popupNtrainClass,'Value')) = val;
            end
            set(edit, 'string', num2str(val));
        end
    end
    function editPw_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= C(data_type_copy)
        elseif sum(answer(1:end)) > 1
        else
            p_w_copy = [answer(1:end-1) 1 - sum(answer(1:end-1))]';
        end
        set(edit, 'string', num2str(p_w_copy'));
    end
    function editTheta_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer < 0
        else
            theta_copy = answer;
        end
        set(edit, 'string', num2str(theta_copy));
    end
    function editC_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= [C(data_type) C(data_type)]
        elseif any(answer<0)
        else
            C_matr_copy = answer;
        end
        set(edit, 'string', num2str(C_matr_copy));
    end
    function editN2C_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 2
        elseif answer(1)==answer(2) || isempty(find(answer(2)==1:C(data_type), 1)) || isempty(find(answer(2)==1:C(data_type), 1))
        else
            n_class_2_copy = answer;
            dataPR2_copy.params.svm_need_retrain = 1;
            dataPR2_copy.params.net_need_retrain = 1;
        end
        set(edit, 'string', num2str(n_class_2_copy));
    end
    function editposC_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif isempty(find(answer==1:2, 1))
        else
            pos_class_copy = ceil(answer);
        end
        set(edit, 'string', num2str(pos_class_copy));
    end
    function editNCompPar_callback(edit, events)
        answer = str2num(get(edit,'string')); change = 0;
        if length(answer) ~= 1
        elseif answer <= 0
        else
            val = ceil(answer); change = 1;
        end
        if change
            if nCompPar_data_copy{1} == 1
                nCompPar_data_copy{2} = val;
            else
                nCompPar_data_copy{3}(get(popupNCompParClass,'Value')) = val;
            end
            set(edit, 'string', num2str(val));
        end
    end
    function editRegPar_callback(edit, events)
        answer = str2num(get(edit,'string')); change = 0;
        if length(answer) ~= 1
        elseif answer < 0
        else
            val = (answer); change = 1;
        end
        if change
            if gmRegularize_data_copy{1} == 1
                gmRegularize_data_copy{2} = val;
            else
                gmRegularize_data_copy{3}(get(popupNCompParClass,'Value')) = val;
            end
            set(edit, 'string', num2str(val));
        end
    end
    function editReplicates_callback(edit, events)
        answer = str2num(get(edit,'string')); change = 0;
        if length(answer) ~= 1
        elseif answer <= 0
        else
            val = ceil(answer); change = 1;
        end
        if change
            if replicates_data_copy{1} == 1
                replicates_data_copy{2} = val;
            else
                replicates_data_copy{3}(get(popupNCompParClass,'Value')) = val;
            end
            set(edit, 'string', num2str(val));
        end
    end

    function popupAlgStart_callback(edit, events)
        val = get(popupAlgStart, 'Value');
        if algStart_data_copy{1} == 1
            algStart_data_copy{2} = val;
        else
            algStart_data_copy{3}(get(popupAlgStart, 'Value')) = val;
        end
        popups_callback;
    end

    function popupCovType_callback(edit, events)
        val = get(popupCovType, 'Value');
        if covType_data_copy{1} == 1
            covType_data_copy{2} = val;
        else
            covType_data_copy{3}(get(popupCovType ,'Value')) = val;
        end
        popups_callback;
    end

    function checkboxSharedCov_callback(edit, events)
        val = get(checkboxSharedCov, 'Value');
        
        if sharedCov_data_copy{1} == 1
            sharedCov_data_copy{2} = val;
        else
            sharedCov_data_copy {3}(get(popupParDistrTypeClass,'Value')) = val;
        end
    end


    function popupIsNaiveBayesType_callback(edit, events)
        val = get(popupIsNaiveBayesType, 'Value');
        if isNaiveBayes_data_copy{1} == 1
            isNaiveBayes_data_copy{2} = val;
        else
            isNaiveBayes_data_copy{3}(get(popupIsNaiveBayesTypeClass,'Value')) = val;
        end
        popups_callback;
    end
    function popupNormType_callback(edit, events)
        val = get(popupNormType, 'Value');
        if normType_data_copy{1} == 1
            normType_data_copy{2} = val;
        else
            normType_data_copy{3}(get(popupNormTypeClass,'Value')) = val;
        end
        popups_callback;
    end
    function popupParDistrType_callback(edit, events)
        val = get(popupParDistrType, 'Value');
        if parDistrType_data_copy{1} == 1
            parDistrType_data_copy{2} = val;
        else
            parDistrType_data_copy{3}(get(popupParDistrTypeClass,'Value')) = val;
        end
        popups_callback;
    end
    function popupParzen1Type_callback(edit, events)
        val = get(popupParzen1Type, 'Value');
        if parzenType_data_copy{1} == 1
            parzenType_data_copy{2}(1) = val;
        else
            parzenType_data_copy{3}(1, get(popupParzenClass,'Value')) = val;
        end
        popups_callback;
    end
    function popupParzen2Type_callback(edit, events)
        val = get(popupParzen2Type, 'Value');
        if parzenType_data_copy{1} == 1
            parzenType_data_copy{2}(2) = val;
        else
            parzenType_data_copy{3}(2, get(popupParzenClass,'Value')) = val;
        end
        popups_callback;
    end

    function editSmoothParX1_callback(edit, events)
        answer = str2num(get(edit,'string')); change = 0;
        if length(answer) ~= 1
        elseif answer <= 0
        else
            val = answer; change = 1;
        end
        if change
            if smoothPar_data_copy{1} == 1
                smoothPar_data_copy{2}(1) = val;
            else
                smoothPar_data_copy{3}(1, get(popupSmoothParClass,'Value')) = val;
            end
            set(edit, 'string', num2str(val));
        end
    end
    function editSmoothParX2_callback(edit, events)
        answer = str2num(get(edit,'string')); change = 0;
        if length(answer) ~= 1
        elseif answer <= 0
        else
            val = answer; change = 1;
        end
        if change
            if smoothPar_data_copy{1} == 1
                smoothPar_data_copy{2}(2) = val;
            else
                smoothPar_data_copy{3}(2, get(popupSmoothParClass,'Value')) = val;
            end
            set(edit, 'string', num2str(val));
        end
    end
    function popupSmoothParCalcAlg_callback(edit, events)
        val = get(edit, 'Value');
        if smoothParCalc_data_copy{1} == 1
            smoothParCalc_data_copy{2}(1) = val;
        else
            smoothParCalc_data_copy{3}(1, get(popupSmoothParCalcClass,'Value')) = val;
        end
        popups_callback;
    end
    function popupSmoothParCalcFeatures_callback(edit, events)
        val = get(edit, 'Value');
        if smoothParCalc_data_copy{1} == 1
            smoothParCalc_data_copy{2}(2) = val;
        else
            smoothParCalc_data_copy{3}(2, get(popupSmoothParCalcClass,'Value')) = val;
        end
        popups_callback;
    end
    function btnSmooParCalcOK_callback(edit, events)
        typeAlg = get(popupSmoothParCalcAlg, 'Value');
        typeFeatures = get(popupSmoothParCalcFeatures, 'Value');
        h_diap = {};
        if typeAlg == 3
            if typeFeatures == 1
                prompt = {'Enter h1 range:','Enter h2 range:'};
                dlg_title = 'Range:';
                num_lines = 1;
                defaultans = {'10.^[-3:.5:3]','10.^[-3:.5:3]'};
                while 1
                    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
                    if isempty(answer)
                        return;
                    end
                    isError = 0;
                    for i = 1:2
                        defaultans{i} = answer{i};
                        h_diap{i} = str2num(defaultans{i});
                        if isempty(h_diap{i}) 
                            prompt{i} = sprintf('Error! Enter correct h%d range:', i);
                            isError = 1;
                            continue;
                        else
                            prompt{i} = sprintf('Enter h%d range:', i);
                        end
                    end
                    if isError == 0
                        break;
                    end
                end
            elseif typeFeatures == 2 || typeFeatures == 3
                i_var = typeFeatures - 1; i_fixed = 3 - i_var;
                prompt = { sprintf('h%d - fixed, enter h%d range:', i_fixed, i_var) };
                dlg_title = 'Range:';
                num_lines = 1;
                defaultans = {'10.^[-3:.1:3]'};
                
                while 1
                    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
                    if isempty(answer)
                        return;
                    end
                    isError = 0;
                    defaultans = answer{1};
                    h_diap = str2num(defaultans);
                    if isempty(h_diap) 
                        prompt{i} = sprintf('Error! Enter correct h%d range:', i);
                        isError = 1;
                    end
                    if isError == 0
                        break;
                    end
                end
            end
        end
        
        if smoothParCalc_data_copy{1} == 1
            % Посчитать сглаживающие параметры для всех классов и
            % одинаковыми для разных классов
            xtrain = dataPR_copy.dataTrain;
            hi = CalcSmoothPars(xtrain, typeAlg, typeFeatures, parzenType_data_copy{2}, smoothPar_data_copy{2}, h_diap);
            if typeFeatures == 1
                smoothPar_data_copy{2}(1) = hi(1);
                smoothPar_data_copy{2}(2) = hi(2);
            elseif typeFeatures == 2
                smoothPar_data_copy{2}(1) = hi(1);
            elseif typeFeatures == 3
                smoothPar_data_copy{2}(2) = hi(2);
            end
        elseif smoothParCalc_data_copy{1} == 0
            type1 = get(popupSmoothParCalcClass, 'Value');
            if type1 == 1
                % Посчитать сглаживающие параметры для всех классов и
                % разными для разных классов
                for c = 1:C(data_type)
                    xtrain = dataPR_copy.dataTrain(dataPR_copy.groupsTrain == c,:);
                    hi = CalcSmoothPars(xtrain, typeAlg, typeFeatures, parzenType_data_copy{3}(:,c), smoothPar_data_copy{3}(:,c), h_diap);
                    if typeFeatures == 1
                        smoothPar_data_copy{3}(1,c) = hi(1);
                        smoothPar_data_copy{3}(2,c) = hi(2);
                    elseif typeFeatures == 2
                        smoothPar_data_copy{3}(1,c) = hi(1);
                    elseif typeFeatures == 3
                        smoothPar_data_copy{3}(2,c) = hi(2);
                    end
                end
            else
                % Посчитать сглаживающие параметры для одного класса
                classN = type1 - 1;
                xtrain = dataPR_copy.dataTrain(dataPR_copy.groupsTrain == classN,:);
                hi = CalcSmoothPars(xtrain, typeAlg, typeFeatures, parzenType_data_copy{3}(:,classN), smoothPar_data_copy{3}(:,classN), h_diap);
                if typeFeatures == 1
                    smoothPar_data_copy{3}(1,classN) = hi(1);
                    smoothPar_data_copy{3}(2,classN) = hi(2);
                elseif typeFeatures == 2
                    smoothPar_data_copy{3}(1,classN) = hi(1);
                elseif typeFeatures == 3
                    smoothPar_data_copy{3}(2,classN) = hi(2);
                end
            end
        end
        popups_callback;
        
    end
    function popupKNN_Metric_callback(edit, events)
        val = get(edit, 'Value');
        kNN_MetricType_copy = val;
        popups_callback;
    end
    function popupKNN_BreakTie_callback(edit, events)
        val = get(edit, 'Value');
        kNN_TieAlgType_copy = val;
        popups_callback;
    end
    function popupKNN_WeightsCalcAlg_callback(edit, events)
        val = get(edit, 'Value');
        kNN_kw_CalcAlg_copy = val;
        popups_callback;
    end
    function editKNN_K_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0
        else
            kNN_K_copy = ceil(answer);
        end
        set(edit, 'string', num2str(kNN_K_copy));
    end
    function editKNN_kw_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0
        else
            kNN_kw_copy = answer;
        end
        set(edit, 'string', num2str(kNN_kw_copy));
    end
    function editKNN_Pmink_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0
        else
            kNN_Pmink_copy = answer;
        end
        set(edit, 'string', num2str(kNN_Pmink_copy));
    end
    function btKNN_WeightsCalcOK_callback(edit, events)
    end
    function popupSVM_KernelType_callback(edit, events)
        val = get(edit, 'Value');
        SVM_KernelType_copy = val;
        dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        popups_callback;
    end
    function popupSVM_SolverType_callback(edit, events)
        val = get(edit, 'Value');
        SVM_SolverType_copy = val;
        dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        popups_callback;
    end
    function checkboxSVM_auto_scale_callback(edit, events)
        val = get(edit, 'Value');
        SVM_auto_scale_copy = val;
        dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        popups_callback;
    end
    function edit_SVM_scale_value_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0
        else
            SVM_scale_value_copy = answer;
            dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        end
        set(edit, 'string', num2str(SVM_scale_value_copy));
    end
    function edit_SVM_MLP_params_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 2
        elseif answer(1) <= 0 || answer(2) >= 0
        else
            SVM_MLP_params_copy = answer;
            dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        end
        set(edit, 'string', num2str(SVM_MLP_params_copy));
    end
    function edit_SVM_pol_degree_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0 || answer > 10
        else
            SVM_pol_degree_copy = ceil(answer);
            dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        end
        set(edit, 'string', num2str(SVM_pol_degree_copy));
    end
    function edit_SVM_outlier_fr_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer < 0 || answer > 1
        else
            SVM_outlier_fr_copy = answer;
            dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        end
        set(edit, 'string', num2str(SVM_outlier_fr_copy));
    end
    function editSVM_C_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0 
        else
            SVM_C_copy = answer;
            dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
        end
        set(edit, 'string', num2str(SVM_C_copy));
    end
    % Neural Networks
    function popupNN_ArchType_callback(edit, events)
        val = get(edit, 'Value');
        NN_arch_type_copy = val;
        dataPR_copy.params.net_need_rebuild = 1; dataPR2_copy.params.net_need_rebuild = 1;
        popups_callback;
    end
    function popupNN_OutType_callback(edit, events)
        val = get(edit, 'Value');
        NN_out_type_copy = val;
        dataPR_copy.params.net_need_rebuild = 1; dataPR2_copy.params.net_need_rebuild = 1;
        popups_callback;
    end
    function popupNN_HiddenType_callback(edit, events)
        val = get(edit, 'Value');
        NN_hidden_type_copy = val;
        dataPR_copy.params.net_need_rebuild = 1; dataPR2_copy.params.net_need_rebuild = 1;
        popups_callback;
    end

    function popupNN_train_alg_callback(edit, events)
        val = get(edit, 'Value');
        NN_train_alg_copy = val;
        popups_callback;
    end
    function editNN_hidden_neurons_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) < 1
        elseif any(answer) <= 0
        else
            NN_hidden_neurons_copy = ceil(answer);
            dataPR_copy.params.net_need_rebuild = 1; dataPR2_copy.params.net_need_rebuild = 1;
            dataPR_copy.params.net_need_retrain = 1; dataPR2_copy.params.net_need_retrain = 1;
        end
        set(edit, 'string', num2str(NN_hidden_neurons_copy));
    end

    function editNN_ValidationChecks_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif any(answer) <= 0
        else
            NN_max_fail_copy = ceil(answer);
            dataPR_copy.params.net_need_retrain = 1; dataPR2_copy.params.net_need_retrain = 1;
        end
        set(edit, 'string', num2str(NN_max_fail_copy));
    end

    function popupNN_perf_alg_callback(edit, events)
        val = get(edit, 'Value');
        NN_perf_type_copy = val;
        dataPR_copy.params.net_need_retrain = 1; dataPR2_copy.params.net_need_retrain = 1;
        popups_callback;
    end

    function btnNN_train_start_callback(edit, events)
        if dataPR_copy.params.net_need_rebuild
            dataPR_copy.learners.net = create_neural_network(NN_arch_type_copy, NN_out_type_copy, NN_hidden_type_copy, NN_hidden_neurons_copy);
            % Rebuild network
            dataPR_copy.params.net_need_rebuild = 0;
        end
        % Train network
        dataPR_copy.learners.net = init(dataPR_copy.learners.net);
        dataPR_copy.learners.net = train_neural_network(dataPR_copy.learners.net, dataPR_copy.dataTrain, dataPR_copy.groupsTrain, NN_out_type_copy, NN_perf_type_copy, NN_max_fail_copy,  NN_train_alg_copy);
        dataPR_copy.params.net_need_retrain = 0;
    end
    function btnNN_view_callback(edit, events)
        if ~isempty(dataPR_copy.learners.net)
            if dataPR_copy.params.net_need_rebuild
                dataPR_copy.learners.net = create_neural_network(NN_arch_type_copy, NN_out_type_copy, NN_hidden_type_copy, NN_hidden_neurons_copy);
                % Rebuild network
                dataPR_copy.params.net_need_rebuild = 0;
            end
            
            view(dataPR_copy.learners.net);
        end
    end
    function btnNN_gensim_callback(edit, events)
        if ~isempty(dataPR_copy.learners.net)
            gensim(dataPR_copy.learners.net);
        end
    end
    % Decision Trees
    function checkboxTree_MergeLeaves_callback(edit, events)
        val = get(edit, 'Value');
        Tree_MergeLeaves_copy = val;
        dataPR_copy.params.tree_need_retrain = 1; dataPR2_copy.params.tree_need_retrain = 1;
        popups_callback;
    end
    function btnTree_train_start_callback(edit, events)
        %dataPR_copy.learners.tree = ...
        dataPR_copy.learners.tree = train_decision_tree(dataPR_copy.dataTrain, dataPR_copy.groupsTrain, Tree_MaxNumSplits_copy, Tree_MinLeafSize_copy, Tree_MinParentSize_copy, Tree_MergeLeaves_copy);
        dataPR_copy.params.tree_need_retrain = 0;
    end
    function btnTree_view_graph_callback(edit, events)
        if ~isempty(dataPR_copy.learners.tree)
            view(dataPR_copy.learners.tree, 'Mode', 'graph')
        end
    end
    function editTree_MaxNumSplits_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0
        else
            Tree_MaxNumSplits_copy = ceil(answer);
            dataPR_copy.params.tree_need_retrain = 1; dataPR2_copy.params.tree_need_retrain = 1;
        end
        set(edit, 'string', num2str(Tree_MaxNumSplits_copy));
    end
    function editTree_MinParentSize_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0
        else
            Tree_MinParentSize_copy = ceil(answer);
            dataPR_copy.params.tree_need_retrain = 1; dataPR2_copy.params.tree_need_retrain = 1;
        end
        set(edit, 'string', num2str(Tree_MinParentSize_copy));
    end
    function editTree_MinLeafSize_callback(edit, events)
        answer = str2num(get(edit,'string'));
        if length(answer) ~= 1
        elseif answer <= 0
        else
            Tree_MinLeafSize_copy = ceil(answer);
            dataPR_copy.params.tree_need_retrain = 1; dataPR2_copy.params.tree_need_retrain = 1;
        end
        set(edit, 'string', num2str(Tree_MinLeafSize_copy));
    end

%% functions shared callbacks
    function popups_callback(cb, event)
        on_off = {'on', 'off'};
        if ~get(checkboxNAll, 'Value')
            set(editN, 'String', num2str(N_data_copy{3}(get(popupNClass,'Value'))));
        else
            set(editN, 'String', num2str(N_data_copy{2}));
        end
        if ~get(checkboxNtrainAll, 'Value')
            set(editNtrain, 'String', num2str(Ntrain_data_copy{3}(get(popupNtrainClass,'Value'))));
        else
            set(editNtrain, 'String', num2str(Ntrain_data_copy{2}));
        end
        if ~get(checkboxNCompParAll, 'Value')
            set(editNCompPar, 'String', num2str(nCompPar_data_copy{3}(get(popupNCompParClass,'Value'))));
            set(editRegPar , 'String', num2str(gmRegularize_data_copy {3}(get(popupNCompParClass,'Value'))));
            set(editReplicates , 'String', num2str(replicates_data_copy{3}(get(popupNCompParClass,'Value'))));
            set(popupAlgStart , 'Value', (algStart_data_copy{3}(get(popupNCompParClass,'Value'))));
            set(popupCovType , 'Value', (covType_data_copy {3}(get(popupNCompParClass,'Value'))));
            set(checkboxSharedCov, 'Value', sharedCov_data_copy{3}(get(popupNCompParClass,'Value')));
        else
            set(editNCompPar, 'String', nCompPar_data_copy{2});
            set(editRegPar , 'String', num2str(gmRegularize_data_copy {2}));
            set(editReplicates , 'String', num2str(replicates_data_copy{2}));
            set(popupAlgStart , 'Value', (algStart_data_copy {2}));
            set(popupCovType , 'Value', (covType_data_copy {2}));
            set(checkboxSharedCov, 'Value', sharedCov_data_copy{2});
        end

        if ~get(checkboxIsNaiveBayesTypeAll, 'Value')
            set(popupIsNaiveBayesType, 'Value', isNaiveBayes_data_copy{3}(get(popupIsNaiveBayesTypeClass,'Value')));
        else
            set(popupIsNaiveBayesType, 'Value', isNaiveBayes_data_copy{2});
        end
        if ~get(checkboxParDistrTypeAll, 'Value')
            set(popupParDistrType, 'Value', parDistrType_data_copy{3}(get(popupParDistrTypeClass,'Value')));
        else
            set(popupParDistrType, 'Value', parDistrType_data_copy{2});
        end
        if ~get(checkboxNormTypeAll, 'Value')
            set(popupNormType, 'Value', normType_data_copy{3}(get(popupNormTypeClass,'Value')));
        else
            set(popupNormType, 'Value', normType_data_copy{2});
        end
        if ~get(checkboxParzenAll, 'Value')
            set(popupParzen1Type, 'Value', parzenType_data_copy{3}(1,get(popupParzenClass,'Value')));
            set(popupParzen2Type, 'Value', parzenType_data_copy{3}(2, get(popupParzenClass,'Value')));
        else
            set(popupParzen1Type, 'Value', parzenType_data_copy{2}(1));
            set(popupParzen2Type, 'Value', parzenType_data_copy{2}(2));
        end
        if ~get(checkboxSmoothParAll, 'Value')
            set(editSmoothParX1, 'String', num2str(smoothPar_data_copy{3}(1,get(popupSmoothParClass,'Value'))));
            set(editSmoothParX2, 'String', num2str(smoothPar_data_copy{3}(2,get(popupSmoothParClass,'Value'))));
        else
            set(editSmoothParX1, 'String', num2str(smoothPar_data_copy{2}(1)));
            set(editSmoothParX2, 'String', num2str(smoothPar_data_copy{2}(2)));
        end
         if ~get(checkboxSmooParCalcAll, 'Value')
             set(popupSmoothParCalcAlg, 'Value', smoothParCalc_data_copy{3}(1, get(popupSmoothParCalcClass,'Value')));
             set(popupSmoothParCalcFeatures, 'Value', smoothParCalc_data_copy{3}(2, get(popupSmoothParCalcClass,'Value')));
         else
             set(popupSmoothParCalcAlg, 'Value', smoothParCalc_data_copy{2}(1));
             set(popupSmoothParCalcFeatures, 'Value', smoothParCalc_data_copy{2}(2));
         end
        
        % Hide or show controls for par classifier
        val1 = get(popupClassifierType, 'Value') == 2;
        val2 = get(popupClassifierType, 'Value')== 2 && get(popupParDistrType, 'Value')==2;
        val3 = get(popupClassifierType, 'Value') == 3;
        val4 = get(popupClassifierType, 'Value') == 4;
        val41 = get(popupClassifierType, 'Value') == 4 && get(popupKNN_Metric','Value') == 2;
        val42 = get(popupClassifierType, 'Value') == 4 && get(popupKNN_Metric','Value') == 5;
        val5 = get(popupClassifierType, 'Value') == 5;
        val51 = val5 && get(popupSVM_KernelType,'Value')==2;
        val52 = val5 && get(popupSVM_KernelType,'Value')==4;
        val53 = val5 && get(checkboxSVM_auto_scale,'Value')==0;
        val6 = get(popupClassifierType, 'Value') == 6;
        val7 = get(popupClassifierType, 'Value') == 7;
        
        vals = [val1 val2 val3 val3 val3 val4 val41 val42 val5 val51 val52 val53 val6 val7];
        
        checksAll = {checkboxParDistrTypeAll, checkboxNCompParAll, checkboxParzenAll, checkboxSmoothParAll, checkboxSmooParCalcAll,[],[],[],[],[],[],[],[],[]};
        controls1 = {{popupParDistrType, txtposParDistrType },...
            {editNCompPar, txtNCompPar,editRegPar, txtRegPar, txtSharedCov, checkboxSharedCov, txtReplicates, editReplicates, txtAlgStart, popupAlgStart, txtCovType, popupCovType },...
            {popupParzen1Type, popupParzen2Type, txtParzenType, txtParzenX1Type, txtParzenX2Type }, ...
            {editSmoothParX1, editSmoothParX2, txtSmoothPar, txtSmoothParX1, txtSmoothParX2 },...
            {popupSmoothParCalcAlg, popupSmoothParCalcFeatures, btnSmooParCalcOK, txtSmoothParCalc, txtSmoothParCalcAlg, txtSmoothParCalcFeatures },...
            {txtKNNPars1, txtKNN_K, editKNN_K, txtKNN_Metric, popupKNN_Metric, txtKNN_TieRule, popupKNN_BreakTie},...
            {txtKNN_kw, editKNN_kw, txtKNN_WeightsCalcAlg, popupKNN_WeightsCalcAlg, btKNN_WeightsCalcOK},...
            {txtKNN_PMink, editKNN_PMink},...
            {txtSVMPars1, txtSVM_KernelType, popupSVM_KernelType, txtSVM_SolverType, popupSVM_SolverType, txtSVM_C, editSVM_C, txtSVMPars2, txtSVM_auto_scale, checkboxSVM_auto_scale, txtSVM_outlier_fr, edit_SVM_outlier_fr },...
            {txtSVM_pol_degree, edit_SVM_pol_degree },...
            {txtSVM_MLP_params, edit_SVM_MLP_params},...
            {edit_SVM_scale_value},...
            {txtNNPars1, popupNN_ArchType, txtNN_OutType, popupNN_OutType, txtNNPars2, editNN_hidden_neurons, txtNN_train_alg, popupNN_train_alg, btnNN_train_start, btnNN_view, btnNN_gensim, txtNN_HiddenType, popupNN_HiddenType, txtNNValidationChecks, editNN_ValidationChecks, txtNN_perf_alg, popupNN_perf_alg}...
            {txtTree_MaxNumSplits, txtTree_MergeLeaves, txtTree_MinParentSize, txtTree_MinLeafSize, editTree_MaxNumSplits, editTree_MinParentSize, editTree_MinLeafSize, checkboxTree_MergeLeaves, btnTree_train_start, btnTree_view_graph}
            };
        popupsClass = {popupParDistrTypeClass, popupNCompParClass, popupParzenClass, popupSmoothParClass, popupSmoothParCalcClass,[],[],[],[],[],[],[],[],[]};
        
        for i = 1:length(checksAll)
            for j = 1:length(controls1{i})
                set(controls1{i}{j}, 'Visible', on_off{2-vals(i)});
            end
            if ~isempty(checksAll{i})
                set(checksAll{i}, 'Visible', on_off{2-vals(i)});
            end
            if ~isempty(popupsClass{i})
                set(popupsClass{i}, 'Visible', on_off{2-(vals(i))});
            end
            
%            set(checksAll{i}, 'Enable', on_off{2-vals(i)});
%            set(popupsClass{i}, 'Enable', on_off{2-(vals(i) & (get(checksAll{i}, 'Value') ~= 1))});
        end
    end
    function checkboxes_callback(cb, event)
        on_off = {'on', 'off'};
        v = get(checkboxNAll, 'Value'); set(popupNClass, 'Enable', on_off{v+1}); N_data_copy{1} = v;
        v = get(checkboxNtrainAll, 'Value'); set(popupNtrainClass, 'Enable', on_off{v+1}); Ntrain_data_copy{1} = v;
        
        v = get(checkboxNCompParAll, 'Value'); set(popupNCompParClass, 'Enable', on_off{v+1}); 
        nCompPar_data_copy{1} = v; gmRegularize_data_copy{1} = v;
        sharedCov_data_copy{1} = v; replicates_data_copy{1} = v; algStart_data_copy{1} = v; covType_data_copy{1} = v;
        
        v = get(checkboxNormTypeAll, 'Value'); set(popupNormTypeClass, 'Enable', on_off{v+1}); normType_data_copy{1} = v;
        v = get(checkboxIsNaiveBayesTypeAll, 'Value'); set(popupIsNaiveBayesTypeClass, 'Enable', on_off{v+1}); isNaiveBayes_data_copy{1} = v;
        v = get(checkboxParDistrTypeAll, 'Value'); set(popupParDistrTypeClass, 'Enable', on_off{v+1}); parDistrType_data_copy{1} = v;
        
        v = get(checkboxParzenAll, 'Value'); set(popupParzenClass, 'Enable', on_off{v+1}); parzenType_data_copy{1} = v;
        v = get(checkboxSmoothParAll, 'Value'); set(popupSmoothParClass, 'Enable', on_off{v+1}); smoothPar_data_copy{1} = v;
        v = get(checkboxSmooParCalcAll, 'Value'); set(popupSmoothParCalcClass, 'Enable', on_off{v+1}); smoothParCalc_data_copy{1} = v;
        
        popups_callback;
    end
    function popupDataType_callback(edit, events)
        change = 0;
        data_type_new = get(edit,'Value');
        Cnew = C(data_type_new);
        if C(data_type_copy) < Cnew
            change = 1;
            N_data_copy{3}(C(data_type_copy)+1:Cnew) = N_data_copy{3}(C(data_type_copy));
        elseif C(data_type_copy) > Cnew
            change = 2
            N_data_copy{3}(Cnew+1:C(data_type_copy)) = [];
        end
        data_type_copy = data_type_new;
        if change
            C_matr_copy = 1 - eye(Cnew,Cnew);
            p_w_copy = 1/Cnew*ones(Cnew,1);
            set(editC, 'string', num2str(C_matr_copy));
            set(editPw, 'string', num2str(p_w_copy'));
            
            popups_update = {popupNtrainClass, popupNClass, popupIsNaiveBayesTypeClass, popupParDistrTypeClass, popupNCompParClass, popupNormTypeClass, popupParzenClass, popupSmoothParClass, popupSmoothParCalcClass};
            types = [1 1 1 1 1 1 1 1 2];
            
            for i = 1:length(popups_update)
                str_i = get_classes_str_arr(types(i));
                set(popups_update{i}, 'String', str_i);
                set(popups_update{i}, 'Value', min([get(popups_update{i}, 'Value') length(str_i) ]));
            end
        end
    end
    function ok_callback(dlg,event)
        ApplyAllData;
        delete(gcf);
    end
    function btnApply_callback(dlg,event)
        ApplyAllData;
    end
    function btnSave_callback(dlg,event)
        bayes_type_copy = get(popupBayesType, 'Value');
        classType_copy = get(popupClassifierType, 'Value');

        vars = {'N_data_copy', 'Ntrain_data_copy', 'p_w_copy', 'theta_copy', 'C_matr_copy', 'n_class_2_copy', 'pos_class_copy', 'bayes_type_copy', 'classType_copy', 'data_type_copy',...
            'isNaiveBayes_data_copy', 'parDistrType_data_copy', 'nCompPar_data_copy', 'gmRegularize_data_copy', 'sharedCov_data_copy', 'replicates_data_copy', 'algStart_data_copy', 'covType_data_copy', 'normType_data_copy', 'parzenType_data_copy', 'smoothPar_data_copy', 'smoothParCalc_data_copy',...
            'kNN_K_copy', 'kNN_MetricType_copy', 'kNN_TieAlgType_copy', 'kNN_kw_copy', 'kNN_kw_CalcAlg_copy', 'kNN_Pmink_copy',...
            'SVM_KernelType_copy', 'SVM_SolverType_copy', 'SVM_C_copy', 'SVM_outlier_fr_copy', 'SVM_pol_degree_copy', 'SVM_MLP_params_copy', 'SVM_auto_scale_copy', 'SVM_scale_value',...
            'NN_arch_type_copy', 'NN_out_type_copy', 'NN_train_alg_copy', 'NN_hidden_neurons_copy', 'NN_hidden_type_copy', 'NN_perf_type_copy', 'NN_max_fail_copy',...
            'Tree_MaxNumSplits', 'Tree_MinLeafSize', 'Tree_MinParentSize', 'Tree_MergeLeaves',...
            'dataPR_copy', 'dataPR2_copy' };
        filename = 'bayes_data_1.mat';
        uisave(vars, filename);
        % todo - deltaT2 - вынести в меню
    end
    function btnLoad_callback(dlg,event)
        bayes_type_copy = 1;
        classType_copy = 1;
        uiopen('load');
        set(popupBayesType, 'Value', bayes_type_copy);
        set(popupClassifierType, 'Value', classType_copy);
        checkboxes_callback;
        popups_callback;
    end
    function ApplyAllData
        change_test = 0; change_train = 0;
        if data_type_copy ~= data_type
            change_test = 1; change_train = 1;
            dataPR_copy.params.net_need_retrain = 1; dataPR2_copy.params.net_need_retrain = 1;
            dataPR_copy.params.net_need_rebuild = 1; dataPR2_copy.params.net_need_rebuild = 1;
            if C(data_type_copy) ~= C(data_type)
                parDistrType_data_copy{3}  = parDistrType_data_copy{2} * ones(1,C(data_type_copy));
                nCompPar_data_copy{3}  = nCompPar_data_copy{2} * ones(1,C(data_type_copy));
                gmRegularize_data_copy{3} = gmRegularize_data_copy{2} * ones(1,C(data_type_copy));
                sharedCov_data_copy{3} = sharedCov_data_copy{2} * ones(1,C(data_type_copy));
                replicates_data_copy{3} = replicates_data_copy{2} * ones(1,C(data_type_copy));
                algStart_data_copy{3} = algStart_data_copy{2} * ones(1,C(data_type_copy));
                covType_data_copy{3} = covType_data_copy{2} * ones(1,C(data_type_copy));
                
                
                isNaiveBayes_data_copy{3} = isNaiveBayes_data_copy{2}  * ones(1,C(data_type_copy));
                normType_data_copy{3} = normType_data_copy{2}  * ones(1,C(data_type_copy));
                parzenType_data_copy{3} = parzenType_data_copy{2} * ones(1,C(data_type_copy));
                smoothPar_data_copy{3} = smoothPar_data_copy{2} * ones(1,C(data_type_copy));
                smoothParCalc_data_copy{3} = smoothParCalc_data_copy{2} * ones(1,C(data_type_copy));
                N_data_copy{3} = N_data_copy{2}* ones(1,C(data_type_copy));
                Ntrain_data_copy{3} = Ntrain_data_copy{2} * ones(1,C(data_type_copy));
            end
        else
            N_new = get_N(N_data_copy, data_type);
            Ntrain_new = get_N(Ntrain_data_copy, data_type);
            if length(N_new) ~= length(get_N(N_data,data_type))
                change_test = 1;
            elseif sum(abs(N_new - get_N(N_data,data_type))) ~= 0
                change_test = 1;
            elseif length(Ntrain_new) ~= length(get_N(nTrain_data,data_type))
                change_train = 1;
            elseif sum(abs(Ntrain_new - get_N(nTrain_data,data_type))) ~= 0
                change_train = 1;
            end
        end
        if n_class_2_copy ~= n_class_2
            [dataPR2_copy.dataAll, dataPR2_copy.groupsAllReal] = GenerateData(data_type_copy, get_N(N_data_copy,data_type_copy), n_class_2_copy);
            [dataPR2_copy.dataTrain, dataPR2_copy.groupsTrain] = GenerateData(data_type_copy, get_N(Ntrain_data_copy,data_type_copy), n_class_2_copy);
        end
        
        if change_test
            [dataPR_copy.dataAll, dataPR_copy.groupsAllReal] = GenerateData(data_type_copy, get_N(N_data_copy,data_type_copy), 1:C(data_type_copy));
            [dataPR2_copy.dataAll, dataPR2_copy.groupsAllReal] = GenerateData(data_type_copy, get_N(N_data_copy,data_type_copy), n_class_2_copy);
        end
        if change_train
            dataPR_copy.params.net_need_retrain = 1; dataPR2_copy.params.net_need_retrain = 1;
            dataPR_copy.params.svm_need_retrain = 1; dataPR2_copy.params.svm_need_retrain = 1;
            [dataPR_copy.dataTrain, dataPR_copy.groupsTrain] = GenerateData(data_type_copy, get_N(Ntrain_data_copy,data_type_copy), 1:C(data_type_copy));
            [dataPR2_copy.dataTrain, dataPR2_copy.groupsTrain] = GenerateData(data_type_copy, get_N(Ntrain_data_copy,data_type_copy), n_class_2_copy);
        end
        
        N_data = N_data_copy;
        nTrain_data = Ntrain_data_copy;
        p_w = p_w_copy;
        theta = theta_copy;
        C_matr = C_matr_copy;
        n_class_2 = n_class_2_copy;
        pos_class = pos_class_copy;
        bayes_type = get(popupBayesType, 'Value');
        data_type = data_type_copy;
        classType = get(popupClassifierType, 'Value');
        isNaiveBayes_data = isNaiveBayes_data_copy;
        parDistrType_data = parDistrType_data_copy;
        nCompPar_data = nCompPar_data_copy;
        gmRegularize_data = gmRegularize_data_copy;
        sharedCov_data = sharedCov_data_copy;
        replicates_data = replicates_data_copy;
        algStart_data = algStart_data_copy; 
        covType_data = covType_data_copy;

        
        normType_data = normType_data_copy;
        parzenType_data = parzenType_data_copy;
        smoothPar_data = smoothPar_data_copy;
        smoothParCalc_data = smoothParCalc_data_copy;
        
        kNN_K = kNN_K_copy;
        kNN_MetricType = kNN_MetricType_copy;
        kNN_TieAlgType = kNN_TieAlgType_copy;
        kNN_kw = kNN_kw_copy;
        kNN_kw_CalcAlg = kNN_kw_CalcAlg_copy;
        kNN_Pmink = kNN_Pmink_copy;
        
        SVM_KernelType = SVM_KernelType_copy;
        SVM_SolverType = SVM_SolverType_copy;
        SVM_C = SVM_C_copy;
        SVM_outlier_fr = SVM_outlier_fr_copy;
        SVM_pol_degree = SVM_pol_degree_copy;
        SVM_MLP_params = SVM_MLP_params_copy;
        SVM_auto_scale = SVM_auto_scale_copy;
        SVM_scale_value = SVM_scale_value_copy;
        
        NN_arch_type = NN_arch_type_copy;
        NN_out_type = NN_out_type_copy;
        NN_hidden_type = NN_hidden_type_copy;
        NN_perf_type = NN_perf_type_copy;
        NN_max_fail = NN_max_fail_copy;
        
        NN_train_alg = NN_train_alg_copy;
        NN_hidden_neurons = NN_hidden_neurons_copy;
        
        Tree_MaxNumSplits = Tree_MaxNumSplits_copy;
        Tree_MinLeafSize = Tree_MinLeafSize_copy;
        Tree_MinParentSize = Tree_MinParentSize_copy;
        Tree_MergeLeaves = Tree_MergeLeaves_copy;
        
        dataPR = dataPR_copy;
        dataPR2 = dataPR2_copy;
    end
%% functions - get strings
    function str_ar = get_classes_str_arr(type)
        Ccur = C(data_type_copy);
        if type == 1
            str_ar = cell(Ccur,1);
            iDiap = 1:Ccur;
        elseif type == 2
            str_ar = cell(Ccur+1,1);
            str_ar{1} = 'All classes';
            iDiap = 2:Ccur+1;
        end
        for i = 1:length(iDiap)
            str_ar{iDiap(i)} = num2str(i);
        end
    end
    function str_ar = get_classifier_types
        str_ar = {'Bayessian Classifier', 'Parametric Estimation', 'Kernel Density Estimation', 'k Nearest Neigbros', 'Support Vector Machines', 'Neural Networks', 'Decision Tree'};
    end
    function str_ar = get_parzen_types
        str_ar = {'Square', 'Gauss', 'Triangle', 'Exp (Laplace)', 'Koshi', 'Regen Filt' };
    end
    function str_ar = get_alg_start
        str_ar = {'randSample', 'plus' };
    end
    function str_ar = get_cov_types
        str_ar = {'full', 'diagonal' };
    end
    function str_ar = get_metric_types
        str_ar = {'euclidean', 'seuclidean', 'cityblock', 'chebychev', 'minkowski', 'mahalanobis', 'cosine', 'correlation', 'spearman', 'hamming', 'jaccard' };
    end
    function str_ar = get_break_tie_rules
        str_ar = {'nearest', 'random', 'consensus' };
    end
    function str_ar = get_SVM_kernel_types
        str_ar = {'linear', 'poly', 'rbf', 'mlp' };
    end
    function str_ar = get_SVM_solver_types
        str_ar = {'SMO', 'ISDA', 'L1QP' };
    end
    function str_ar = get_NN_arch_types
        str_ar = {'ff', 'cascade'};
    end
    function str_ar = get_NN_out_types
        str_ar = {'tansig', 'softmax', 'satlin', 'logsig', 'purelin'};
    end
    function str_ar = get_NN_hidden_types
        str_ar = {'tansig', 'logsig', 'poslin', 'satlin', 'satlins', 'purelin'};
    end
    function str_ar = get_NN_perf_types
        str_ar = {'mae', 'mse', 'sae', 'sse', 'crossentropy'};
    end
    function str_ar = get_NN_train_types
        str_ar = { 'traingd', 'traingda', 'traingdm', 'traingdx', 'trainrp', 'traincgf', 'traincgb', 'traincgp', 'trainscg', 'trainlm', 'trainbfg', 'trainoss', 'trainbr'};
    end
end
%% Other Dialogs
function choice = choosedialog(items, name, answer, val)

    d = dialog('Position',[300 300 250 150],'Name',name);
    txt = uicontrol('Parent',d,...
           'Style','text',...
           'Position',[20 80 210 40],...
           'String',answer);
       
    popup = uicontrol('Parent',d,...
           'Style','popup',...
           'Position',[75 70 100 25],...
           'String',items,...
            'Value', val);
       
    btn = uicontrol('Parent',d,...
           'Position',[40 20 70 25],...
           'String','OK',...
           'Callback', @ok_callback);
       
    choice = val;
       
    % Wait for d to close before running to completion
    uiwait(d);
    
        function ok_callback(dlg,event)
        choice = popup.Value;
        delete(gcf)
        end
end