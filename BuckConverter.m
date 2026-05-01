Vref = 30;              % Desired output voltage (V)
L = 2e-3;               % Inductance (H)
C = 220e-6;             % Capacitance (F)
R = 15;                 % Load resistance (ohm)
fs = 20e3;              % Switching frequency (Hz)
Ts = 1/fs;              % Switching period (s)

Tend = 0.10;            % Total simulation time (s)
dt = 1e-5;              % Simulation step for averaged model
t = 0:dt:Tend;          % Time vector


Vin = 60 - 20*(t/Tend);


Ntrain = 12000;

Vo_train = 20 + 20*rand(1, Ntrain);            % Output voltage samples in [20, 40]
Vin_train = 40 + 20*rand(1, Ntrain);           % Input voltage samples in [40, 60]
e_train = Vref - Vo_train;                     % Error
de_train = -2 + 4*rand(1, Ntrain);             % Change in error in [-2, 2]

% Ideal duty cycle + correction for dynamic action
D_ideal = Vref ./ Vin_train;
D_target = D_ideal + 0.015*e_train + 0.01*de_train;

% Saturate target duty cycle to physical range
D_target = max(0.05, min(0.95, D_target));

X = [e_train; de_train; Vin_train];
T = D_target;

%% -----------------------------
% 4. Train neural network
%% -----------------------------
hiddenLayerSize = 12;
net = fitnet(hiddenLayerSize, 'trainlm');

net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

net.trainParam.epochs = 300;
net.trainParam.goal = 1e-6;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.1;
net.divideParam.testRatio  = 0.1;

net = train(net, X, T);

iL = zeros(size(t));
Vo = zeros(size(t));
D  = zeros(size(t));

Vo(1) = 0;
iL(1) = 0;
e_prev = Vref - Vo(1);

for k = 1:length(t)-1
    e = Vref - Vo(k);
    de = e - e_prev;

    % Neural network controller
    D(k) = net([e; de; Vin(k)]);

    % Saturation of duty cycle
    D(k) = max(0.05, min(0.95, D(k)));

    % Averaged state equations
    diL = (D(k)*Vin(k) - Vo(k))/L;
    dVo = (iL(k) - Vo(k)/R)/C;

    % Euler integration
    iL(k+1) = iL(k) + dt*diL;
    Vo(k+1) = Vo(k) + dt*dVo;

    e_prev = e;
end

D(end) = D(end-1);

%% ----------------------------------------
% 6. Performance measurements
%% ----------------------------------------
ss_index = round(0.02/dt):length(t);   % Ignore startup, assess after 20 ms
Vo_ss = Vo(ss_index);

mean_Vo = mean(Vo_ss);
max_dev = max(abs(Vo_ss - Vref));
rmse = sqrt(mean((Vo_ss - Vref).^2));

fprintf('Mean output voltage after startup = %.4f V\n', mean_Vo);
fprintf('Maximum absolute deviation       = %.4f V\n', max_dev);
fprintf('RMSE                             = %.4f V\n', rmse);

%% -----------------------------
% 7. Plots
%% -----------------------------
figure('Color','w');

subplot(3,1,1);
plot(t, Vin, 'LineWidth', 1.8);
grid on;
ylabel('Input Voltage (V)');
title('Input Voltage');

subplot(3,1,2);
plot(t, Vo, 'b', 'LineWidth', 1.8); hold on;
yline(Vref, 'r--', 'LineWidth', 1.5);
grid on;
ylabel('Output Voltage (V)');
title('Output Voltage Regulation');
legend('V_o', 'V_{ref}=30V', 'Location', 'best');

subplot(3,1,3);
plot(t, D, 'm', 'LineWidth', 1.8);
grid on;
xlabel('Time (s)');
ylabel('Duty Cycle');
title('Control Signal');

%% ----------------------------------------
% 8. Optional: save data for Simulink use
%% ----------------------------------------
simData.t = t(:);
simData.Vin = Vin(:);
simData.Vo = Vo(:);
simData.D = D(:);
save('buck_nn_results.mat', 'simData', 'net', 'L', 'C', 'R', 'Vref', 'fs', 'Ts');

%% ------------------------------------------------------------
% 9. Parameters for MATLAB Function block inside Simulink
%% ------------------------------------------------------------
% To use the trained network in Simulink, extract weights/biases:
IW = net.IW{1,1};
b1 = net.b{1};
LW = net.LW{2,1};
b2 = net.b{2};

save('buck_nn_controller.mat', 'IW', 'b1', 'LW', 'b2');

disp('Files saved: buck_nn_results.mat, buck_nn_controller.mat');