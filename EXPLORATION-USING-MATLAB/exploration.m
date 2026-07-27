data=load('.\S1.mat\S1.mat');
% Display all variable names inside the file
disp(fieldnames(data));

% Display the size of the main EEG variable
% (we'll confirm the exact variable name once you run fieldnames)
disp(size(data.data));  % <-- replace 'data' with the actual field name shown above

trial = data.data(:, :, 7, 2);  % size: 64 x 150
size(trial)

% Load frequency/phase info
freqinfo = load('Freq_Phase.mat');

% Display variable names inside
disp(fieldnames(freqinfo));

disp(size(freqinfo.freqs));
disp(size(freqinfo.phases));

% disp(freqinfo.freqs);
% disp(freqinfo.phases);

% Open and read the .loc file as plain text
fid = fopen('64-channels.loc', 'r');
locData = textscan(fid, '%d %f %f %s');
fclose(fid);

% Display first few rows
disp(locData{1}(1:5));  % channel numbers
disp(locData{2}(1:5));  % angle 1 (theta)
disp(locData{3}(1:5));  % angle 2 (radius)
disp(locData{4}(1:5));  % channel labels (names like 'Oz', 'Pz')

