# EEG and ECOG
Code base and summary surrounding my thesis on simultaneous EEG and ECOG brain recordings.


![image](https://user-images.githubusercontent.com/44694489/213349390-abbe68de-d49b-4a9b-bab5-a242d335f372.png)

ECoG signals are useful for evaluating BSS algorithms because the signal to noise ratios of ECoG signals are higher than those of EEG signals. The non ECoG correlated components in the EEG signals must be artefacts. Thus, extractions of a BSS component similar to ECoG closely equates to extraction of neural components from EEG signals. Each EEG component was evaluated by measuring how much information the component shared with the ECoG signals. If it worked, its component was highly correlated or not correlated at all with the ECoG signals. 

5 minutes of 4 experiments; being (Low Anaesthetic, Deep Anaesthetic, Rest and Recovery)  of simultaneous ECoG and EEG data was provided by the NeuroTycho dataset [23].  ECOG locations were provided as a two-dimensional position grid.

EEG positions were provided as a text file which refers to the 10-20 system. This is an internationally recognized method which describes the location of scalp electrodes. Unfortunately, three-dimensional data points for ECOG was not available so the EEG positions were dimensionally reduced to the XZ plane.  This was done by taking the X and Z data points from the 10-20 positions, and then shifting and scaling which was required to fit the 10-20 EEG points to the ECOG data points

![image](https://user-images.githubusercontent.com/44694489/213349712-4c7e5b43-ed8e-4fa9-a7e6-088256046fff.png)

Once the brain signal EEG data was pre-processed to an acceptable standard. The feature selection stage began
Work began programming RNN’s and CNN’s in python. There were many challenges to this as most of the architecture was suited to other data. An attempt was made to adapt a time series predictive gated recursive neural network to the BSS problem.


Correlations can be used to facilitate a process known as feature extraction to filter highly correlated data which provides no further information to the neural network

![image](https://user-images.githubusercontent.com/44694489/213352059-cd3d83a8-4627-4bb0-adda-80ceb7c12486.png)

Correlations between the EEG and ECoG channels were also computed.



