%EEG preproccesing matlab script
%import the EEG data into workspace 

%takes the spectrum of the EEG n channel
spectrogram(EEG(1,:),[],[],[],1000,'yaxis');

%specify line frequency noise, at 50hz and Q factor for notch filter. 
wo = 50/(1000/2);  bw = wo/35;
[b,a] = iirnotch(wo,bw,200);
Y = filter(b,a,EEG(:,:));
spectrogram(Y(1,:),[],[],[],1000,'yaxis')
figure 
spectrogram(EEG(1,:),[],[],[],1000,'yaxis')

% Standard EEG filter cut offs Low frequency filter: 1 Hz High frequency filter: 50-70 Hz
%Implements a high pass butterworth 5th order, we know low frequencies we
%do not care about and contain no information

[z,p,k] = butter(5,300/500,'high');
sos = zp2sos(z,p,k);
