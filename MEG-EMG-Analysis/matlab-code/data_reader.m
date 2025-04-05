%%%%%% QUESTIONS %%%%%%
%% Which are the channels for the accelerometer?
%%  For the EEG, which channels do i have to subtract?
%% show my_reader.m


% uses library lvm_import_3p12

addpath(genpath('./lvm_import_3p12'))

clear('all');
direc = '/Users/federicobonato/Developer/WORK/data/plfp65/';
fn_codes = 'plfp65_rec7_13.11.2024_13-42-47_array1.lvm';

data = lvm_import([ direc char(fn_codes)],2);

n_pts = size(data.Segment1.data,1)

fs = abs(round(1/(data.Segment1.data(3,1)-data.Segment1.data(4,1))))

time = [1:n_pts] / fs;

figure
for i= 1:20 
    subplot(6,10,i);
    plot(time,data.Segment1.data(:,i+1)); 
    ylim([-3000 3000]);
    title(['MEG Ch ' num2str(i) ' X'])
end
for i= 1:20 
    subplot(6,10,i+20);
    plot(time,data.Segment1.data(:,i+64+1)); 
    ylim([-3000 3000]); 
    title(['MEG Ch ' num2str(i) ' Y'])
end
for i= 1:20 
    subplot(6,10,i+40);
    plot(time,data.Segment1.data(:,i+128+1)); 
    ylim([-3000 3000]); 
    title(['MEG Ch ' num2str(i) ' Z'])
end

figure
for i= 1:32 
    subplot(6,6,i);
    plot(time,data.Segment1.data(:,i+192+1)); 
    % ylim([-3000 3000]); 
   title(['Other Ch ' num2str(i+192+1)])
end

figure
plot(time,data.Segment1.data(:,27+192+1));



