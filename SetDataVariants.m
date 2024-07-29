%% Задание вариантов исходных распределений классов
function class_data = SetDataVariants
global data_type_names axis_rect data_type;
global data_type_per_class C;
% type=1 - нормальное распределение
% par1 - средние значения (строка [c1 c2 ... cn]
% par2 - ковариацоинная матрица

% type=2 - гауссова смесь
% par1 - матрица средних значений компонент (каждая строка - вектор средних [c1 c2 ... cn]
% par2 - 3-мерный массив с ковариацоинными матрицами компонент
% par3 - вектор с коэффициентами pi каждой компоненты

% type=3 - равномерное распределение
% par1 - массив ячеек, каждая из которых характеризует одну из областей,
% где расположено равномерное распределение - {area1, area2, ..., areaN}
% area - описание области в формате {type, par_area1, par_area2, ...}
% par2 - то же, что par1, но описывает области, которные необходимо
% исключить
% Далее приведены примеры описания различных областей
% {1, [x1min x1max x2min x2max], angle} - прямоугольник с точками {x1min, x2min} и {x1max, x2max},
% повернутый относительно центра на угол angle
% {2, [с1 с2], r} - окружность с центром в [c1 c2] радиуса r
% {3, [с1 с2], [r1 r2]} - эллипс с центром в [c1 c2], радиусами r1 и r2,
% повернутый относительно центра на угол angle
% {4, [p11 p12], [p21 p22], [p31 p32], angle} - треугольник с координатами
% вершин {p11, p12}, {p21, p22}, {p31, p32}, повернутый относительно центра
% на угол angle
% {5, xv, yv, angle} - внутренняя часть полигона, заданного точками {xv(i),
% yv(i)}, повернутого относительно центра на угол angle

% type =4 - распределение точек вокруг кривой на плоскости по заданному
% закону (положение на кривой - центр, имеет равномерное распределение
% par1 - описание кривой в форме {curvePar1, curvePar2, ..., curveParN}
% curvePar - описывает часть кривой - возможны следующие варианты
% {1, tstart, tfinish, @fx(t), @fy(t)} - описание в параметрической форме
% x=fx(t), y=fy(t), tstart < t < tfinish
% {2, xv, yv} - описание в форме последовательности координат точек
% {xv{i},yv{i}}
% par2 - описание распределения вокруг кривой в формате {distPar1,
% distPar2, ..., distparN}
% distPar - описывает распределение точек вокруг соответствующей кривой,
% заданной curvePar
% {1, [s1 s2 r]} - нормальное распределение с СКО s1, s2 и к-том корреляции r
% {2, [r1 r2]} - равмномерное распределение в прямоугольнике, отстоящем на
% r1 вправо-влево, r2 - вверх-вниз
% {3, r} - равмномерное распределение в круге радиуса r
% par3 - коэффициенты (вероятности) каждых частей в формате 
% {1} - число точек пропорционально длине кривой
% {2,  [p1 p2 ... pN]} - число точек для каждой кривой задается через pi

data_type = 1; % 1 - нормальное распределение, 2 - смеси, 3 - равномерное
Nmax = 100;
class_data = cell(Nmax,1);
axis_data = cell(Nmax,1);
data_type_names = cell(Nmax,1);

% Классы имеют нормальные распределения

class1.type = 1;
class1.par1 = [3 2];
class1.par2 = eye(2,2);

class2.type = 1;
class2.par1 = [5 5];
class2.par2 = [1 .5; .5 1];

class3.type = 1;
class3.par1 = [4 6];
class3.par2 = eye(2,2);

class_data{1} = { class1, class2, class3 };
axis_data{1} = [-5 10 -2 13];
data_type_names{1} = 'Normal';

% Классы имеют распределения в форме
% GMM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%angle1 = 90; % Угол поворота в градусах
%theta1 = deg2rad(angle1); % Переводим угол в радианы
% Класс 1
class1.type = 2; % Указывает тип GMM
class1.par1 = [-200 300; 0 100; 200 300]; % Параметры для средних значений компонент Гаусса
class1.par2 = cat(3, [400 0; 0 400], [400 0; 0 400], [400 0; 0 400]); % Параметры для ковариационных матриц компонент Гаусса
class1.par3 = ones(1, 3) / 3; % Коэффициенты смешивания компонент Гаусса

% Класс 2
class2.type = 2;
class2.par1 = [-80 200; 80 200; -300 200]; % количество центров распределения 
class2.par2 = cat(3,  [300 0; 0 300], [300 0; 0 300], [300 0; 0 300]); % Параметры для ковариационных матриц компонент Гаусса
class2.par2(:, :, 3) = [270 0; 0 270]; 
class2.par3 = [0.3 0.5 0.2];

% Класс 3
class3.type = 2;
class3.par1 = [300 200; 300 100; -300 100];
class3.par2 = cat(3, [300 0; 0 300], [500 0; 0 500], [300 0; 0 300]);
class3.par2(:, :, 2) = [60 0; 0 60]; % Вторая компонента повернута на 90 градусов
class3.par3 = [0.4 0.3 0.3];

% Класс 4
class4.type = 2;
class4.par1 = [-250 -50; 250 -50; -100 -300];
class4.par2 = cat(3, [400 0; 0 400], [400 0; 0 400], [400 0; 0 400]);
class4.par2(:, :, 2) = [60 0; 0 60]; % Вторая компонента повернута на 90 градусов
class4.par3 = [0.5 0.3 0.2];

% Класс 5
class5.type = 2;
class5.par1 = [0 -400; -150 -200; 150 -200];
class5.par2 = cat(3, [400 0; 0 400], [400 0; 0 400], [400 0; 0 400]);
class5.par2(:, :, 2) = [30 0; 0 30]; % Вторая компонента повернута на 90 градусов
class5.par3 = [0.3 0.6 0.1];

% Класс 6
class6.type = 2;
class6.par1 = [100 -300; 200 -100; 300 0];
class6.par2 = cat(3, [200 0; 0 200], [200 0; 0 200], [200 0; 0 200]);
class6.par2(:, :, 2) = [60 0; 0 60]; % Вторая компонента повернута на 90 градусов
class6.par3 = [0.8 0.1 0.1];

% Класс 7
class7.type = 2;
class7.par1 = [-200 -150; 300 100; -150 250];
class7.par2 = cat(3, [200 0; 0 200], [200 0; 0 200], [200 0; 0 200]);
class7.par2(:, :, 2) = [60 0; 0 60]; % Вторая компонента повернута на 90 градусов
class7.par3 = [0.6 0.2 0.2];

% Класс 8
class8.type = 2;
class8.par1 = [150 250; 220 -100; 300 50];
class8.par2 = cat(3, [200 0; 0 200], [200 0; 0 200], [200 0; 0 200]);
class8.par2(:, :, 2) = [60 0; 0 60]; % Вторая компонента повернута на 90 градусов
class8.par3 = [0.7 0.2 0.1];

% Сохраняем классы в ячейковом массиве
class_data{1} = {class1, class2, class3, class4, class5, class6, class7, class8};

% Задаем данные для осей и имена типов данных
axis_data{1} = [-400 400 -400 400]; % Пределы осей для отображения
data_type_names{1} = 'GMM'; % Имя типа данных

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
% Эллипс 1 (слева)
uni_ellipse1_center1 = {3, [-4 -2], [4 -2], 20}; 
% Эллипс 2 (справа)
uni_ellipse1_center2 = {3, [4 2], [-4 2], 20}; 

% Прямоугольник 1 
uni_rect1_center1 = {1, [-1 1 -6 6], 0}; 

% Прямоугольник 2 
uni_rect2_center1 = {1, [-5 5 -1 1], 0}; 

% Прямоугольник 3 
uni_rect3_center1 = {1, [-4 4 1 3], 0}; 

% Прямоугольник 4 
uni_rect4_center1 = {1, [-3 3 3 5], 0}; 

class1.type = 3;
class1.par1 = {uni_ellipse1_center1};
class1.par2 = {};

class2.type = 3;
class2.par1 = {uni_ellipse1_center2};
class2.par2 = {};

class3.type = 3;
class3.par1 = {uni_rect1_center1};
class3.par2 = {};

class4.type = 3;
class4.par1 = {uni_rect2_center1};
class4.par2 = {};

class5.type = 3;
class5.par1 = {uni_rect3_center1};
class5.par2 = {};

class6.type = 3;
class6.par1 = {uni_rect4_center1};
class6.par2 = {};

% Сохраняем классы в ячейковом массиве
class_data{3} = {class1, class2, class3, class4, class5, class6};

% Задаем данные для осей и имена типов данных
axis_data{3} = [-7 7 -7 7];
data_type_names{3} = 'Uniform';
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1



















% Классы имеют различные распределения (смеси, нормальное, равномерное)

class1.type = 1;
class1.par1 = [1 2];
class1.par2 = eye(2,2);

class2.type = 2;
class2.par1 = [5 -1;-3 6];
class2.par2 = cat(3,[3 0;0 .5],[3 0;0 3]);
class2.par3 = [0.2 0.7];

class3.type = 3;
class3.par1 = {{4, [-7 0], [-6 4], [-4 2], 15},  {2, [5 5], 3}, {3, [5 -5], [4 1], -20} };
class3.par2 = {};

class4.type = 3;
class4.par1 = {{2, [4 10], 2}, {1, [-7 0 -6 0], 20}, {5, [6 7 8 6 8 6], [-4 -1 -4 -2 -2 -4], -10}};
class4.par2 = {{1,[-5 -2 -5 -1], 20}};

class_data{4} = { class1, class2, class3, class4 };
axis_data{4} = [-8 10 -8 13];
data_type_names{4} = 'Norm+GMM+Uniform';

% Классы имеют равномерные распределения в форме концентрических
% окружностей
class1.type = 3;
class1.par1 = {{2, [2 2], 2}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{2, [2 2], 4}};
class2.par2 = {{2, [2 2], 2}};

class3.type = 3;
class3.par1 = {{2, [2 2], 6}};
class3.par2 = {{2, [2 2], 4}};

class_data{5} = { class3, class2, class1 };
axis_data{5} = [-8 10 -8 13];
data_type_names{5} = 'Conc. Circles';

% Классы имеют равномерные распределения в форме горизонтальных полос
class1.type = 3;
class1.par1 = {{1, [1 7 1 3], 0}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{1, [1 7 3 5], 0}};
class2.par2 = {};

class3.type = 3;
class3.par1 = {{1, [1 7 5 7], 0}};
class3.par2 = {};

class_data{6} = { class1, class2, class3 };
axis_data{6} = [0 8 0 8];
data_type_names{6} = 'Hor. Stripes';


% Классы имеют равномерные распределения в форме вертикальных полос
class1.type = 3;
class1.par1 = {{1, [1 3 1 7], 0}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{1, [3 5 1 7], 0}};
class2.par2 = {};

class3.type = 3;
class3.par1 = {{1, [ 5 7 1 7], 0}};
class3.par2 = {};

class_data{7} = { class1, class2, class3 };
axis_data{7} = [0 8 0 8];
data_type_names{7} = 'Ver. Stripes';

% Классы имеют равномерные распределения в форме 4 квадратов
class1.type = 3;
class1.par1 = {{1, [1 3 1 3], 0}};
class1.par2 = {};

class2.type = 3;
class2.par1 = {{1, [3 5 1 3], 0}};
class2.par2 = {};

class3.type = 3;
class3.par1 = {{1, [ 1 3 3 5], 0}};
class3.par2 = {};

class4.type = 3;
class4.par1 = {{1, [ 3 5 3 5], 0}};
class4.par2 = {};

class_data{8} = { class1, class2, class3, class4 };
axis_data{8} = [0 6 0 6];
data_type_names{8} = 'Squares';

% Классы имеют равномерные распределения в форме 4 квадратов
circle_data = {{2, [0 0], 1}};
p = { [-2 -2],[0 -2],[2 -2], [2 0],[2 2],[0 2],[-2 2],[-2 0],[0 0] }; 

class.type = 3;
class.par1 = circle_data;
class.par2 = {};

class_data{9} = { class };
for i = 1:length(p) - 1
    class.type = 3;
    if i == length(p)-1
        class.par1 = {{4, p{i}, p{1}, p{length(p)},0}};
    else
        class.par1 = {{4, p{i}, p{i+1}, p{length(p)},0}};
    end;        
    class.par2 = circle_data;
    class_data{9} = [class_data{9}, class ];
end;
axis_data{9} = [-3 3 -3 3];
data_type_names{9} = 'Circle+Lines';

% Задание в форме точек, распределенных вокруг кривой
class1.type = 4; 
fx = @(t, k1, k2)t.*cos(k1*t+k2); fy = @(t, k1, k2)t.*sin(k1*t+k2);
tmin = 0; tmax = 5; dt = .2;
k1 = 2; k2 = 0; 
s1 = .14; s2 = .14; r = 0;
class1.par1 = {{2, fx(tmin:dt:tmax, k1, k2), fy(tmin:dt:tmax, k1, k2)}};  % {2, xv, yv}
class1.par2 = {{1, [s1 s2 r]}};     % {1, [s1 s2 r]} - нормальное распределение с СКО s1, s2 и к-том корреляции r
class1.par3 = {1};

class2.type = 4; 
k2 = 2;
r = 0.2;
class2.par1 = {{1, tmin, tmax, @(x)fx(x,k1,k2), @(x)fy(x,k1,k2)}}; % {1, tstart, tfinish, @fx(t), @fy(t)}
class2.par2 =  {{3, r}}; % равмномерное распределение в круге радиуса r
class2.par3 = {1};

class3.type = 4; 
k2 = 4;
r1 = 0.25; r2 = 0.25;
class3.par1 = {{1, tmin, tmax, @(x)fx(x,k1,k2), @(x)fy(x,k1,k2)}}; % {1, tstart, tfinish, @fx(t), @fy(t)}
class3.par2 = {{2, [r1 r2]}};     % {2, [r1 r2]} - равмномерное распределение в прямоугольнике, отстоящем на
class3.par3 = {1};

class_data{10} = {class1, class2, class3};
axis_data{10} = [-5 5 -5 5];
data_type_names{10} = 'Spiral';

% Один из вариантов мозайки

h = 1;
p_start = [0; 0];
n1 = 6;
n2 = 6;
hex_pgm = generate_hex_pgm(h, p_start, n1, n2);
[ncl, t_pgm] = tile_hex_pgm (hex_pgm, 1);
class_data{11} = {};

for c = 1:ncl
    class.type = 3;
    class.par1 = {};
    i_c = find(t_pgm == c);
    for i = 1:length(i_c)
        ii = i_c(i);
        class.par1{i} = {5, hex_pgm{ii}(1,:), hex_pgm{ii}(2,:), 0}; %{5, xv, yv, angle}
    end;
    class.par2 = {};
    class_data{11} = [class_data{11}, class ];
end;
axis_data{11} = [0 6 0 6];
data_type_names{11} = 'Hex.Mozaic';

plot_temp = 0;
if plot_temp
    colors = {'r','g','b','y'};
    for i = 1:n1
        for j = 1:n2
            plot(hex_pgm{i,j}(1,:), hex_pgm{i,j}(2,:));
            patch(hex_pgm{i,j}(1,:),hex_pgm{i,j}(2,:),colors{t_pgm(i,j)});
            hold on;
        end;
    end;
    axis([0 h*sqrt(3)*(n1-1) 0 3*h*(n2-1)/2]);
end;

class_data = class_data(~cellfun('isempty',class_data));
axis_data = axis_data(~cellfun('isempty',axis_data));
data_type_names = data_type_names(~cellfun('isempty',data_type_names));

axis_rect = axis_data;

data_type_per_class = cell(length(class_data),1);
C = zeros(1, length(class_data));
for i = 1:length(class_data)
    C(i) = length(class_data{i});
    for j = 1:length(class_data{i})
        data_type_per_class{i}(j) = class_data{i}{j}.type;
    end;
end;


% Данные, описывающие рапределение исходных классов
%class_data = {class_data1, class_data2, class_data3, class_data4, class_data5, class_data6, class_data7, class_data8, class_data9, class_data10};

end
function hex_poly_matr = generate_hex_pgm(h, p_start, n1, n2)
k = h*sqrt(3)/2;
hex_poly = [0 k k 0 -k -k 0;-h -h/2 h/2 h h/2 -h/2 -h];
hex_poly = hex_poly + p_start * ones(1, size(hex_poly,2));

hex_poly_matr = cell(n1, n2);

for i = 1:n1
    for j = 1:n2
        hex_poly_matr{i,j} = hex_poly + [(i-1)*2*k + mod(j-1,2) * k;(j-1)*3/2*h] * ones(1, size(hex_poly,2));
    end;
end;

end
function [ncl, t_pgm] = tile_hex_pgm (pgm, type)
[n1,n2] = size(pgm);
t_pgm = zeros(n1,n2);
ncl = 4;

f_ind = @(i,j)(mod(j+1,2)==0&&mod(j+1,4)~=0 && mod(i+1,2)==0) || (mod(j+1,4) == 0 && mod(i,2)==0);

for i = 1:n1
    for j = 1:n2
        t_pgm(i,j) =    [1:4]*[f_ind(i,j);f_ind(i+1,j); f_ind(i,j+1); f_ind(i+1,j+1)];
    end;
end;

end
