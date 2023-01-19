# EEG and ECOG
Code base and summary surrounding my thesis on simultaneous EEG and ECOG brain recordings.

MATLAB and Python were used for the preprocessing and feature extraction platforms, and the following toolboxes were used  to run various functions: 

●	EEG Lab toolbox, which includes the SOBI, Infomax and JADE algorithms

●	Signal Processing toolbox, which includes filtering functions and Spectrogram  

●	Statistics toolbox, which included histogram and pdf functions.

●	Python libraries including Tensorflow and Scikit Learn. which included the FastICA algorithm 


![image](https://user-images.githubusercontent.com/44694489/213349390-abbe68de-d49b-4a9b-bab5-a242d335f372.png)

ECoG signals are useful for evaluating Blind Source Seperation algorithms because the signal to noise ratios of ECoG signals are higher than those of EEG signals. The non ECoG correlated components in the EEG signals must be artefacts. Thus, extractions of a BSS component similar to ECoG closely equates to extraction of neural components from EEG signals. Each EEG component was evaluated by measuring how much information the component shared with the ECoG signals. If it worked, its component was highly correlated or not correlated at all with the ECoG signals. 

5 minutes of 4 experiments; being (Low Anaesthetic, Deep Anaesthetic, Rest and Recovery)  of simultaneous ECoG and EEG data was provided by the NeuroTycho dataset [23].  ECOG locations were provided as a two-dimensional position grid.

EEG positions were provided as a text file which refers to the 10-20 system. This is an internationally recognized method which describes the location of scalp electrodes. Unfortunately, three-dimensional data points for ECOG was not available so the EEG positions were dimensionally reduced to the XZ plane.  This was done by taking the X and Z data points from the 10-20 positions, and then shifting and scaling which was required to fit the 10-20 EEG points to the ECOG data points.

Seen in;
- electrode_position_mapping.m

![image](https://user-images.githubusercontent.com/44694489/213349712-4c7e5b43-ed8e-4fa9-a7e6-088256046fff.png)

Once the brain signal EEG data was pre-processed to an acceptable standard using fourier transform, normalisation, baseline removal and filtering 
- EEGFilterAllChannels.m
- EEG_Preprocessing.m

Work began programming a recursive neural network in python to predict one of the EEG sensors data given the multimodal data. There were many challenges to this as most of the architecture was suited to other data. An attempt was made to adapt a time series predictive gated recursive neural network to the BSS problem seen in; 

- ECoG_RNN_Predict.py
- EEG_RNN_Predict.py
- ECoG_RNN_Predict_withEEG.py

With the following example predictions;
![ECoG_Prediction_Results_100_samples](https://user-images.githubusercontent.com/44694489/213357069-de84ebd6-49ad-4b83-9d29-1d65cd7a39f9.png)
![EEG_Prediction_Results_24_sample_pred](https://user-images.githubusercontent.com/44694489/213357527-4016a9e1-e2bb-4cca-83b5-117cb883f3d0.png)


Correlations can be used to facilitate a process known as feature extraction to filter highly correlated data which provides no further information to the neural network. Correlations between the EEG and ECoG channels were then computed and finally the correlations of the outputs of each of the current well known blind source seperation algorithms was computed as a measure of performance.  




![image](https://user-images.githubusercontent.com/44694489/213352059-cd3d83a8-4627-4bb0-adda-80ceb7c12486.png)

Standard pre-processing was applied in order to filter the EEG signals and remove noise. Once this had occurred, a variety of algorithms that are responsible for performing BSS were applied to the pre-processed EEG signals and the Pearson correlation coefficients and histograms between the outputs of the algorithms and the raw ECoG signals were measured. 

An example can be seen in; 
- Compute ICA.ipynb
- EEGandECOG_matlab_correlations.m

![image](https://user-images.githubusercontent.com/44694489/213358949-ac556e4f-a8ec-410b-a3ad-9915182f7def.png)



