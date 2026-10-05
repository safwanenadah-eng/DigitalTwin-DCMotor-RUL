%% Script de Génération de Dataset pour la Prédiction du RUL
clear; clc;

% 1. Charger les paramètres nominaux du moteur
init_params;

% 2. Configuration du scénario de dégradation
num_cycles = 100;                 % Nombre de cycles de fonctionnement
b_max = 5 * b_nominal;            % Usure maximale (5x le frottement nominal)

% Profil d'augmentation progressive du frottement (Usure des roulements)
b_degradation = linspace(b_nominal, b_max, num_cycles);

% Nom de votre fichier Simulink 
model_name = 'dc_motor_twin';    

% Préparation du chargement du modèle
load_system(model_name);

% Matrices pour stocker les données d'apprentissage
dataset_current = [];
dataset_speed   = [];
time_vector     = [];
dataset_rul     = [];

fprintf('--> Début de la génération automatisée (%d cycles)...\n', num_cycles);

% 3. Boucle de simulation automatisée
for k = 1:num_cycles
    % Injection du paramètre usé dans le Workspace pour Simulink
    b = b_degradation(k);
    
    % Calcul du RUL (Remaining Useful Life) normalisé de 1.0 (100%) à 0.0 (0%)
    rul = (num_cycles - k) / (num_cycles - 1);
    
    % Exécution de la simulation Simulink sur 10 secondes (ou t_end)
    simOut = sim(model_name, 'StopTime', '10');
    
    % Extraction des signaux enregistrés par le Scope / Outports
    % (Ajustez les indices selon la structure de votre objet out/simOut)
    current_signal = simOut.yout{1}.Values.Data;
    speed_signal   = simOut.yout{2}.Values.Data;
    t              = simOut.yout{1}.Values.Time;
    
    % Sauvegarde dans nos matrices
    if k == 1
        time_vector = t; % On garde le vecteur temps de référence
    end
    
    dataset_current = [dataset_current, current_signal];
    dataset_speed   = [dataset_speed, speed_signal];
    dataset_rul     = [dataset_rul; rul];
    
    % Affichage de la progression dans la Command Window
    fprintf('Cycle %3d/%d | b = %.6f | RUL = %5.1f%%\n', k, num_cycles, b, rul * 100);
end

% 4. Sauvegarde du jeu de données pour Python
if ~exist('../data', 'dir')
    mkdir('../data');
end
save('../data/motor_degradation_dataset.mat', 'dataset_current', 'dataset_speed', 'time_vector', 'b_degradation', 'dataset_rul');

fprintf('\n--> Génération terminée avec succès ! Dataset enregistré dans data/motor_degradation_dataset.mat\n');