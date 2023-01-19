

%X and Y
% starting brain values
T = 1:128; 
T = zeros(1,128)
T(1) = 0; % first ECoG electrode channel data point 
T2 = 1:8;
T2 = zeros(1,8)
T2(1) = 0; 
load('ChibiMAP.mat')
lh = figure;
image(I);axis equal
hold on
%relocating the postions on the EEG original EEG 10/20 positions 
X2 = [0.0949; 0.0586; 0.0677; 0; 0; -0.0677; -0.0949; -0.0586]+0.0949
Y2 = [-0.0047; -0.0088; 0.0469; -0.0104; 0.0668; 0.0469; -0.0047; -0.0088]+0.0104 
% Make the first frame: 
% scatter plot parameters, x, y, circle size,  T*sin(t(1)/2)
h = scatter(X,Y,100,T,'filled'); 
po = scatter(X2*4400+100, -Y2*4400+950,200, T2,'filled')
% set x/y axis limits: 
axis([0 max(X) 0 max(Y)]) 
% set color axis limits: 
%colormap([jet(20);parula(64)])
caxis([-4000 8000]) 
cb = colorbar; 
ylabel(cb,'voltage') 


%set EEG vector to reference relevent EEG channels. 
EEG_ref_v = [1,3,4,8,9,12,13,15];
%reads data from excel
%EEG = csvread('EEG_rest.csv');
for ii =  1:8; 
    set(1,'cdata',2*ECoG(ii,100))
    EEG_ref_v(ii)
    pause(0.000000000000000000001)
end
%scales EEG values to be visualized in colour map. 
%will need to use this again whenever you change states of monkey
%EEGMeans = transpose(mean(transpose(EEG)));  Means without filtering 
axis tight manual

