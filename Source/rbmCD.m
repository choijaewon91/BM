function [ W, b, c, e, samples ] = rbmCD( visible_node, num_hidden, mu, size_batch, tot_iter, num_gibbstep, save_freq, printout)
%% Initialize internal parameters
num_visible = size(visible_node,2);
num_data = size(visible_node,1);

W=0.1.*randn(num_visible,num_hidden);
b=zeros(1,num_visible);
c=zeros(1,num_hidden);

num_batch = ceil(num_data/size_batch);

T=linspace(0,1,num_Temp);

h_m=cell(num_Temp,1);
v_m=cell(num_Temp,1);

%% Mix Data Set
mixed_visible_node = visible_node(randperm(num_data),:);

%% Divide data set into minibatches
batch=cell(num_batch,1);
for i=1:num_batch
    if(i==num_batch && mod(num_data,size_batch)~=0 )
        batch{i}=mixed_visible_node( (i-1)*size_batch+1 : end , : );
    else
        batch{i}=mixed_visible_node( (i-1)*size_batch+1 : i*size_batch , : );
    end
    
end

if(printout==1)
    disp(['# of visible node:   ' num2str(num_visible)]);
    disp(['# of hidden  node:   ' num2str(num_hidden)]);
end

curr_iter=0;
%% Training RBM
for i=1:tot_iter
    for j=1:num_batch
        curr_iter=curr_iter+1;
        
        v0=batch{j};
        num_curr_batch=size(v0,1);
        
        %% positive gradient
        h0=1./(1+exp(-bsxfun(@plus,v0*W, c)));
        v_data=mean(v0,1);
        h_data=mean(h0,1);
        vh_data=v0'*h0./num_curr_batch;

        h_r=binornd(1, h0, num_curr_batch, num_hidden);
        v_r=1./(1+exp(-bsxfun(@plus,h_r*W',b)));
        
        e(curr_iter)=mean(sum((v0-v_r).^2,2));
        
        %% Contrastive Divergence
        h_m=h0;
        num_data_model = size(h0,1);
        % Sampling Step
        for n = 1:num_gibbstep
            hard_h=binornd(1, h_m, num_data_model, num_hidden);
            v_m=1./(1 + exp(bsxfun(@plus,hard_h*W',b) ) );
            h_m=1./(1 + exp(bsxfun(@plus,v_m*W,c) ) );
        end
        
        %% negative gradient
        v_model=mean(v_m{end},1);
        h_model=mean(h_m{end},1);
        vh_model=v_m{end}'*h_m{end}./num_data_model;
        
       %% update parameters
       W=W+mu.*(vh_data-vh_model);
       b=b+mu.*(v_data-v_model);
       c=c+mu.*(h_data-h_model);
       samples = v_m{end};
       %% decreasing update parameter
       mu=mu;
       
       
       %% Save parameters and output to file
       if(mod(curr_iter,save_freq)==0)
            if(printout)
                disp(['Saving to File - Current Iteration:  ' num2str(curr_iter)]);
            end
            if(exist('filename'))
                if(printout)
                    disp(['Deleting File:  ' filename]);
                end
                filename = [filename '.mat']; 
                delete(filename);
            end
            today=date;
            filename = ['rbmCD_' num2str(curr_iter) '_' today]; 
            if(printout)
                disp(['Savining File:  ' filename]);
            end
            save(filename, 'W','b','c','e','num_hidden','mu','size_batch', 'tot_iter', 'num_gibbstep', 'num_Temp', 'swap_iter');
        end
       
    end
end






end

