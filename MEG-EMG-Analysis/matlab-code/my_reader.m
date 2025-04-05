%%%%%% QUESTIONS %%%%%%
%% Which are the channels for the accelerometer?
%%  For the EEG, which channels do i have to subtract?

% uses library lvm_import_3p12

addpath(genpath('./lvm_import_3p12'))

clear('all');
direc = '/Users/federicobonato/Developer/WORK/lid_opm/data/plfp65/';
fn_codes = 'plfp65_rec4_13.11.2024_13-17-33_array1.lvm';

data = lvm_import([ direc char(fn_codes)],2);

n_pts = size(data.Segment1.data,1);

fs = abs(round(1/(data.Segment1.data(3,1)-data.Segment1.data(4,1))));

time = [1:n_pts] / fs;


% Figure 1: MEG coordinate X
figure;
tiledlayout(4,5);
for i= 1:20 
    nexttile;
    plot(time,data.Segment1.data(:,i+1)); 
    ylim([-3000 3000]);
    title(['MEG Ch ' num2str(i) ' X']);
end


% Figure 2: MEG coordinate Y
figure;
tiledlayout(4,5);
for i= 1:20 
    nexttile;
    plot(time,data.Segment1.data(:,i+64+1)); 
    ylim([-3000 3000]); 
    title(['MEG Ch ' num2str(i) ' Y']);
end

% Figure 3: MEG coordinate Z
figure;
tiledlayout(4,5);
for i= 1:20 
    nexttile;
    plot(time,data.Segment1.data(:,i+128+1)); 
    ylim([-3000 3000]); 
    title(['MEG Ch ' num2str(i) ' Z']);
end

% Figure 4: Trigger Channel
figure;
plot(time, data.Segment1.data(:,end-4)); % Using the index of MUX_Counter1 (trigger)
title('Trigger Channel');


% Figure 5: Accelerometer Data (T1 to T11)

figure;
tiledlayout(4,3); % Layout 4x3 for visualizing all channels

for i = 1:11
    nexttile;
    plot(time, data.Segment1.data(:, 192 + i)); % T1 corresponds to column 193
    title(['T' num2str(i)]);
end

%  Figure 6: PLOT FOR EEG (A1 - A16)

figure;
tiledlayout(4,4);
for i = 1:16
    nexttile;
    plot(time, data.Segment1.data(:, 196 + i)); % A1-A16
    title(['EEG Unipolar A' num2str(i)]);
end

% Figure 7: Single channel (example)
figure;
plot(time, data.Segment1.data(:,27+192+1)); % An example of a single channel
title(['Other Ch ' num2str(i+192+1)]);


