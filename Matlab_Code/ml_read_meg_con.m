% T. Sander, PTB, Berlin, 2024
% tilmann.sander-thoemmes@ptb.de

% needs fieldtrip

clear;

ft_defaults;

dir = 'dir';
fn_codes = { 'name';};

fn = [ char( fn_codes ) '.con']; 
hdr = ft_read_header([dir fn ]);

cfg             = [];
cfg.datafile    = [dir fn];
cfg.headerfile  = [dir fn];
cfg.continuous  = 'yes';
cfg.Fs          = hdr.Fs;
cfg.nSamples 	= hdr.nSamples;
cfg.channel 	= {'all'};
data            = ft_preprocessing(cfg);

plot(data.trial{1}(32,:))