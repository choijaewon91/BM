%% main function

%% Presetting
addpath('MNIST_DATA','Result','Source');
images = loadMNISTImages('train-images.idx3-ubyte');
images=images';
%% parameters
visible_node = images;
num_hidden = 100;
mu = 0.00007;
size_batch = 1000;
tot_iter = 10^5;
num_gibbstep = 1;
num_Temp = 21;
swap_iter = 2;
save_freq = 1e2;
printout = 1;
update_rate=[0.9 1 1.1];

%% machine to run

runRBM=0;
runGRBM=1;
%%
if(runRBM)
	[W, b, c, e] = rbmPT( visible_node, num_hidden, mu, size_batch, tot_iter, num_gibbstep, num_Temp, swap_iter, save_freq, printout);
end

if(runGRBM)
	visible_node = 255.*visible_node;
	[W, b, c, e] = grbmPT( visible_node, num_hidden, mu, size_batch, tot_iter, num_gibbstep, num_Temp, swap_iter, save_freq, printout,update_rate);
end