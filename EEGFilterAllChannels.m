Fs = 1000;
filtCoeff = designfilt('bandstopiir', 'FilterOrder', 2, 'HalfPowerFrequency1', 49, 'HalfPowerFrequency2', 51, 'SampleRate', 1000);
fvtool(filtCoeff);

for i = 1:16
%LineFreeEEG_Recovery(i,:) = filter(filtCoeff, EEG(i,:));
figure
subplot(2,1,1)
periodogram(EEG(i,:), [], [], Fs);
subplot(2,1,2)
periodogram(LineFreeEEG_Rest(i,:), [],[],Fs);
%must still compensate for filter delay
%periodogram(ECoG(1,:),[],[], Fs); 
end
%{
figure
plot(EEG(1,1:300))
hold on
plot(LineFreeEEG_Recovery(1,1:300))
%}
grpdelay(filtCoeff,512,1000)