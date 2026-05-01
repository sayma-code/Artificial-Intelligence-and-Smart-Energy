%% Fuzzy Logic for Transformer Thermal Condition (Updated for new MATLAB)

% Create Mamdani FIS
fis = mamfis('Name','TransformerThermalCondition');

%% 1. INPUT VARIABLES

% 1) Load (%)
fis = addInput(fis,[0 150],'Name','Load');
fis = addMF(fis,'Load','trapmf',[0 0 40 70],'Name','Low');
fis = addMF(fis,'Load','trimf',[40 80 120],'Name','Medium');
fis = addMF(fis,'Load','trapmf',[90 130 150 150],'Name','High');

% 2) Oil Temperature (°C)
fis = addInput(fis,[40 110],'Name','OilTemp');
fis = addMF(fis,'OilTemp','trapmf',[40 40 50 70],'Name','Low');
fis = addMF(fis,'OilTemp','trimf',[50 75 90],'Name','Medium');
fis = addMF(fis,'OilTemp','trapmf',[75 100 110 110],'Name','High');

% 3) Ambient Temperature (°C)
fis = addInput(fis,[0 45],'Name','AmbTemp');
fis = addMF(fis,'AmbTemp','trapmf',[0 0 10 20],'Name','Low');
fis = addMF(fis,'AmbTemp','trimf',[10 25 35],'Name','Medium');
fis = addMF(fis,'AmbTemp','trapmf',[25 40 45 45],'Name','High');

%% 2. OUTPUT VARIABLE

fis = addOutput(fis,[0 10],'Name','Condition');
fis = addMF(fis,'Condition','trimf',[0 2 4],'Name','Normal');
fis = addMF(fis,'Condition','trimf',[3 5 7],'Name','Warning');
fis = addMF(fis,'Condition','trimf',[6 8 10],'Name','Critical');

%% 3. RULE BASE (Updated Syntax)

% Rule format:
% "IF Load==Low & OilTemp==Low & AmbTemp==Low THEN Condition==Normal"
% Use string-based rules in new MATLAB

ruleList = [
    "Load==Low & OilTemp==Low & AmbTemp==Low => Condition=Normal"
    "Load==Medium & OilTemp==Medium & AmbTemp==Medium => Condition=Warning"
    "Load==High => Condition=Critical"
    "OilTemp==High => Condition=Critical"
];

fis = addRule(fis,ruleList);

%% 4. Evaluate the system for the given inputs

inputValues = [85 75 30];   % [Load, OilTemp, AmbTemp]

ConditionValue = evalfis(fis,inputValues);

fprintf('Crisp Thermal Condition Output: %.3f\n', ConditionValue);

% Interpretation
if ConditionValue < 3.5
    disp('Thermal Condition: NORMAL');
elseif ConditionValue < 6.5
    disp('Thermal Condition: WARNING');
else
    disp('Thermal Condition: CRITICAL');
end

%% Optional: Plot membership functions
% figure; plotmf(fis,'input',1);
% figure; plotmf(fis,'input',2);
% figure; plotmf(fis,'input',3);
% figure; plotmf(fis,'output',1);
