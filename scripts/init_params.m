%% Script d'initialisation des paramètres - Jumeau Numérique Moteur DC
clear; clc;

%% 1. Paramètres Électriques
R_nominal = 2.0;       % Résistance statorique nominale (Ohm)
L_stator  = 0.005;     % Inductance statorique (Henry)
Ke        = 0.05;      % Constante de force contre-électromotrice (V / (rad/s))
U_supply  = 24.0;      % Tension d'alimentation continue (Volts)

%% 2. Paramètres Mécaniques
J_total   = 0.002;     % Moment d'inertie total (kg.m^2)
b_nominal = 0.0005;    % Frottement visqueux nominal (N.m.s / rad)
Kt        = 0.05;      % Constante de couple (N.m / A)
T_load    = 0.2;       % Couple de charge constant (N.m)
b = b_nominal;         % Variable lue par le bloc Rotational Damper
%% 3. Paramètres de Simulation & Santé
Ts        = 0.001;     % Pas de temps d'échantillonnage (1 ms)
t_end     = 100;       % Durée d'un cycle de fonctionnement (s)

disp('--> Paramètres physiques chargés avec succès dans le Workspace !');