%% preprocessing_steps.m — show the signal after each Step 8 preprocessing stage
% Pipeline: DC removal -> zero-phase bandpass 6-65 Hz (5th-order Butterworth) -> channel-wise z-score
% No toolbox needed: filter coefficients (sos.mat) come from SciPy.
clear; clc; close all;

% ---- Settings ----
dataDir = '/Users/roshan/projects/gnn/SSVEP-TRIAL-REPO/FIRST_RUNS/EXPLORATION-USING-MATLAB/';
subj = 1;  tgt = 7;  blk = 2;   % MATLAB 1-indexed
chan = 48;                      % Pz (MATLAB index)
fs   = 250;  band = [6 65];

S   = load([dataDir sprintf('S%d.mat', subj)]);
fp  = load([dataDir 'Freq_Phase.mat']);
sosS = load([dataDir 'sos.mat']);  sos = sosS.sos;     % 5 x 6 second-order sections
f0  = fp.freqs(tgt);
X0  = S.data(:, :, tgt, blk);           % 64 x 1500, raw
t   = (-125:1374) / fs;                 % -0.5 s ... 5.5 s (stimulus onset at 0)

%% Stage 1: DC removal (subtract each channel's mean over time)
X1 = X0 - mean(X0, 2);

%% Stage 2: zero-phase bandpass 6-65 Hz (forward + backward SOS filtering)
X2 = sosfiltfilt_nt(X1, sos);

%% Stage 3: channel-wise z-score (per channel, over time, this trial)
X3 = (X2 - mean(X2, 2)) ./ std(X2, 0, 2);

%% Numeric check first (mean / std of the chosen channel at each stage)
stages = {'1 Raw', '2 DC removed', '3 Bandpass 6-65 Hz', '4 Z-scored'};
Xs = {X0, X1, X2, X3};
for i = 1:4
    fprintf('%-20s mean = %10.4f   std = %10.4f\n', stages{i}, mean(Xs{i}(chan,:)), std(Xs{i}(chan,:)));
end

%% Figure 1: time domain (left) and spectrum (right) at each stage
N = size(X0, 2);  f = (0:N/2) * fs / N;
figure('Position', [100 100 1200 800]);
for i = 1:4
    x = Xs{i}(chan, :);

    subplot(4, 2, 2*i-1);
    plot(t, x); xline(0, 'r--'); xlim([t(1) t(end)]); grid on;
    title([stages{i} ' — time']); ylabel('Amplitude');
    if i == 4, xlabel('Time from stimulus onset (s)'); end

    Y = abs(fft(x)) / N;  Y = Y(1:N/2+1);  Y(2:end-1) = 2*Y(2:end-1);
    subplot(4, 2, 2*i);
    semilogy(f, Y + eps); xlim([0 80]); grid on; hold on;
    for h = 1:4, xline(h*f0, 'g--'); end          % f0 and harmonics
    xline(band(1), 'k:'); xline(band(2), 'k:');   % passband edges
    title([stages{i} ' — spectrum']); ylabel('|Y|');
    if i == 4, xlabel('Frequency (Hz)'); end
end
sgtitle(sprintf('S%d, target %d (%.1f Hz), block %d, channel index %d (Oz)', subj, tgt, f0, blk, chan));

%% Figure 2: the filter itself (manual frequency response)
w = linspace(0, 100, 2000);  z = exp(-1j*2*pi*w/fs);  H = ones(size(w));
for s = 1:size(sos, 1)
    H = H .* (sos(s,1) + sos(s,2)*z + sos(s,3)*z.^2) ./ (sos(s,4) + sos(s,5)*z + sos(s,6)*z.^2);
end
figure('Position', [150 150 700 400]);
plot(w, 20*log10(abs(H) + eps)); hold on;
plot(w, 40*log10(abs(H) + eps), '--');
xline(band(1), 'k:'); xline(band(2), 'k:'); xlim([0 100]); ylim([-80 5]); grid on;
legend('single pass', 'forward+backward'); xlabel('Hz'); ylabel('dB');
title('Butterworth 6-65 Hz, order 5');

%% ---- Local function (must stay at the end of the file) ----
function y = sosfiltfilt_nt(x, sos)
% x: channels x time. Odd-extension padding, then forward-backward SOS filtering
% using base MATLAB filter() only.
n   = size(sos, 1);
pad = 3 * (2*n + 1);
x   = x';                                            % time x channels
ext = [2*x(1,:) - x(pad+1:-1:2, :); x; 2*x(end,:) - x(end-1:-1:end-pad, :)];
y = ext;
for pass = 1:2
    for s = 1:n
        y = filter(sos(s,1:3), sos(s,4:6), y);
    end
    y = flipud(y);                                   % reverse for the backward pass
end
y = y(pad+1:end-pad, :)';                            % strip padding, back to channels x time
end