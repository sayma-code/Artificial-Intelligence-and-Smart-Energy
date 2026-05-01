%% 1. Simulation and plant parameters
Tsim   = 0.5;          % total simulation time [s]
Ts     = 1e-4;         % simulation step [s]
t      = 0:Ts:Tsim;    % time vector
N      = numel(t);

Vref   = 50;           % desired output voltage [V]

% Boost converter (averaged model) parameters
L      = 1e-3;         % inductance [H]
C      = 1e-3;         % output capacitance [F]
Rload  = 25;           % load resistance [ohm]

% Duty cycle limits
Dmin   = 0.05;
Dmax   = 0.9;

%% 2. Irradiance profile and PV input voltage model
% Two irradiance levels: high (G1) and low (G2)
G1 = 1000;   % W/m^2
G2 = 400;    % W/m^2

% Step change in irradiance at t_step
t_step = 0.2;  % [s]
G = G1*ones(1,N);
G(t >= t_step) = G2;

% Map irradiance to PV voltage range [20, 40] V
Vpv_min = 20; Vpv_max = 40;
G_min   = G2; G_max   = G1;
Vpv = Vpv_min + (G - G_min)./(G_max - G_min) * (Vpv_max - Vpv_min);
Vpv = max(Vpv_min, min(Vpv_max, Vpv));   % saturation

%% 3. Neural network controller definition (online adaptive)
% Network: 2 inputs (error, Vpv), 1 hidden layer (Nh neurons), 1 output (duty)
Nh = 6;

% Weight initialization
rng(1);
W1 = 0.1*randn(Nh,2);     % input -> hidden
b1 = zeros(Nh,1);
W2 = 0.1*randn(1,Nh);     % hidden -> output
b2 = 0;

eta = 5e-3;               % learning rate

% Activation functions
tanh_act  = @(x) tanh(x);
dtanh_act = @(x) 1 - tanh(x).^2;
sigm      = @(x) 1./(1+exp(-x));
dsigm     = @(x) sigm(x).*(1-sigm(x));

%% 4. Preallocate state and logging variables
iL   = zeros(1,N);   % inductor current
Vo   = zeros(1,N);   % output voltage
D    = zeros(1,N);   % duty cycle
e    = zeros(1,N);   % voltage error

% Initial conditions
iL(1) = 0;
Vo(1) = 30;          % start below reference

%% 5. Main simulation loop (averaged boost + online NN training)
for k = 1:N-1
    % --- Controller inputs ---
    e(k)    = Vref - Vo(k);      % voltage error
    x_in    = [e(k); Vpv(k)];    % 2x1
    
    % --- Forward pass through NN ---
    net1    = W1*x_in + b1;      % Nhx1
    a1      = tanh_act(net1);    % hidden layer
    net2    = W2*a1 + b2;        % scalar
    u_raw   = sigm(net2);        % (0,1)
    
    % Map to duty cycle range
    D(k)    = Dmin + (Dmax - Dmin)*u_raw;
    
    % --- Boost converter averaged model (Euler integration) ---
    % diL/dt = (Vpv - (1-D)*Vo)/L
    diL     = (Vpv(k) - (1 - D(k))*Vo(k))/L;
    % dVo/dt = ((1-D)*iL - Vo/R)/C
    dVo     = ((1 - D(k))*iL(k) - Vo(k)/Rload)/C;
    
    iL(k+1) = iL(k) + Ts*diL;
    Vo(k+1) = Vo(k) + Ts*dVo;
    
    % --- Online weight update (gradient descent on 0.5*e^2) ---
    % Recompute forward pass with current states for gradient (same as above)
    e_k     = Vref - Vo(k+1);    % use next-step error for adaptation
    x_in    = [e_k; Vpv(k)];     % 2x1
    net1    = W1*x_in + b1;
    a1      = tanh_act(net1);
    net2    = W2*a1 + b2;
    u_raw   = sigm(net2);
    
    % Sensitivity of loss wrt net2
    dJ_du   = -(Vref - Vo(k+1));   % d(0.5*e^2)/de * de/du, approximate de/du ≈ -1
    du_dnet2 = dsigm(net2);
    dJ_dnet2 = dJ_du * du_dnet2;
    
    % Gradients for W2, b2
    dJ_dW2  = dJ_dnet2 * a1';
    dJ_db2  = dJ_dnet2;
    
    % Backprop to hidden layer
    dnet2_da1 = W2';
    dJ_da1    = dnet2_da1 * dJ_dnet2;          % Nhx1
    da1_dnet1 = dtanh_act(net1);              % Nhx1
    dJ_dnet1  = dJ_da1 .* da1_dnet1;          % Nhx1
    
    % Gradients for W1, b1
    dJ_dW1    = dJ_dnet1 * x_in';             % Nhx2
    dJ_db1    = dJ_dnet1;
    
    % Weight update
    W2 = W2 - eta*dJ_dW2;
    b2 = b2 - eta*dJ_db2;
    W1 = W1 - eta*dJ_dW1;
    b1 = b1 - eta*dJ_db1;
end

% Last sample error and duty
e(end) = Vref - Vo(end);
D(end) = D(end-1);

%% 6. Plots

figure;
subplot(3,1,1);
plot(t, G, 'LineWidth',1.2);
ylabel('Irradiance G [W/m^2]');
grid on;
title('Irradiance Profile and PV Input Voltage');

yyaxis right;
plot(t, Vpv, 'LineWidth',1.2);
ylabel('V_{pv} [V]');

subplot(3,1,2);
plot(t, Vo, 'LineWidth',1.2);
hold on; yline(Vref,'r--','LineWidth',1.2);
ylabel('V_o [V]');
grid on;
title('Output Voltage Regulation');

subplot(3,1,3);
plot(t, D, 'LineWidth',1.2);
ylabel('Duty cycle D');
xlabel('Time [s]');
grid on;
title('Neural Network Controller Output');

figure;
subplot(2,1,1);
plot(t, iL, 'LineWidth',1.2);
ylabel('i_L [A]');
grid on;
title('Inductor Current');

subplot(2,1,2);
plot(t, e, 'LineWidth',1.2);
ylabel('Voltage error e [V]');
xlabel('Time [s]');
grid on;
title('Control Error vs Time');