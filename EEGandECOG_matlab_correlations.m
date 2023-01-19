%Simultaneous EEG and ECOG Code
%info 
%B. EEG_{experiments}.mat
	%Data matrix: Channel x Time
	%Sampling rate: 1000Hz
	%Location of electrodes: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, C4, T4, T5, P3, T6, O1, O2 (determined by 10-20 system)

%Run EEGlab, 
%import EEG data 
%select default channel locations data comes with EEGlibrary in sample_data
%directory 
%delete respective channels. seen above
%chanlocs = struct('labels', { 'Fp1' 'Fp2' 'F7' 'F3' 'Fz' 'F4' 'F8' 'T3' 'C3' 'C4' 'T4' 'T5' 'P3' 'T6' 'O1' 'O2' }); 
%pop_chanedit( chanlocs );
TotalSamples = 300000; 
SamplingRate = 1000;
TotalTime_S = TotalSamples/SamplingRate; % in seconds
TotalTime_Min = TotalTime_S/60;
time = linspace(0,TotalTime_Min,length(EEG));
plot(time,EEG(1,:))
title('EEG')
xlabel('Time (minutes) ') 
ylabel('Voltage') 
J = 1:5;                                
%Normalizing data only works for 2018!! 
%N = normalize(J);
%/for i = 1:10
%end
%M = max(EEG)
%dlmwrite('EEG_low-anesthetic.txt',M)
%cross correlation matrix 
% cross = xcorr2(a,b) a and b are matrices eeg and ECOG 
%hist(EEG(1,:))
%finds the variance of the EEG channels
EEGT = transp(EEG);
ECoGT = transp(ECoG);
EEG_var =var(EEGT)
%Min/Max Values
Max = 1:128
Min = 1:128;
%find min and max values of each electrode 
for k = 1:128
   Min(k) =  min(ECoG(k,:))
   Max(k) =  max(ECoG(k,:))
end
%find reference electrode 
%
%find correlation coefficients.
CORR = [];
for k = 1:16
    for j = 1:128
    coef = corrcoef(EEGT(:,k),ECoGT(:,j))
    CORR(k,j) = coef(1,2);
    end
end

                                                                                                                                                                                                                                                                        


