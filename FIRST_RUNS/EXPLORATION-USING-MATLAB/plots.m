data=load('.\S1.mat\S1.mat');

fid = fopen('64-channels.loc', 'r');
locData = textscan(fid, '%d %f %f %s');
fclose(fid);

labels = locData{4};
oz_index = find(strcmpi(labels, 'Oz'));
disp(oz_index);

% Choose one trial: target 1 (8.0 Hz), block 1
trial = data.data(:, :, 1, 1);   % shape: 64 x 1500

% Sampling rate
fs = 250;  % Hz
n_samples = size(trial, 2);

% Build a time axis in seconds
time_axis = (0:n_samples-1) / fs;  % 0 to 6 seconds

% We know electrode 61 = Oz (we'll verify this from the .loc file)
oz_channel = 62;

figure;
plot(time_axis, trial(oz_channel, :));
xlabel('Time (seconds)');
ylabel('Voltage (\muV)');
title('Raw EEG - Electrode O1 - Target 1 (8.0 Hz) - Block 1');



fs = 250;
n_samples = size(trial, 2);
time_axis = (0:n_samples-1) / fs;

% Find FP1 and Oz indices from the .loc file
fid = fopen('64-channels.loc', 'r');
locData = textscan(fid, '%d %f %f %s');
fclose(fid);

labels = locData{4};
fp1_channel = find(strcmpi(labels, 'FP1'));
oz_channel  = find(strcmpi(labels, 'Oz'));

disp(['FP1 index: ', num2str(fp1_channel)]);
disp(['Oz index: ', num2str(oz_channel)]);

% Plot both channels stacked in subplots for easy comparison
figure;

subplot(2,1,1);
plot(time_axis, trial(fp1_channel, :), 'r');
xlabel('Time (seconds)');
ylabel('Voltage (\muV)');
title('Raw EEG - Electrode FP1 (Frontal) - Target 1 (8.0 Hz)');

subplot(2,1,2);
plot(time_axis, trial(oz_channel, :), 'b');
xlabel('Time (seconds)');
ylabel('Voltage (\muV)');
title('Raw EEG - Electrode Oz (Occipital) - Target 1 (8.0 Hz)');