function [ W, b, c, e ] = rbmPT( visible_node, num_hidden, mu, size_batch, tot_iter, num_gibbstep, num_Temp, swap_iter, save_freq, printout)
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
    disp('Gaussian-Bernoulli Boltzmann Machine + Parallel Tempering');
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
        if(isnan(e(curr_iter)) )
            disp('Error: e Diverging');
        end
        %% Parallel Tempering
        % Sampling Step
        for n = 1:num_gibbstep
            for m = 1:num_Temp
                %for first iteration initialize h_m
                if (curr_iter==1 && n==1)
                    h_m{m}=h0;
                    num_data_model = size(h0,1);
                end
                hard_h=binornd(1, h_m{m}, num_data_model, num_hidden);
                v_m{m}=1./(1 + exp( -T(m) .* bsxfun(@plus,hard_h*W',b) ) );
                h_m{m}=1./(1 + exp( -T(m) .* bsxfun(@plus,v_m{m}*W,c) ) );
            end           
        end
        % swap every swap_iter
        if(mod(curr_iter, swap_iter)==0)
            if(printout==1)
                disp(['Fantasy Particle Swapping - Current Iteration:  ' num2str(curr_iter)]);
            end
            for m = 1:num_Temp-1
                E0_T0 = -T(m).*(v_m{m}*b.') - sum( log( 1+exp(T(m) .* bsxfun(@plus,v_m{m}*W,c) ) ),2 );
                E0_T1 = -T(m+1).*(v_m{m}*b.') - sum( log( 1+exp(T(m+1) .* bsxfun(@plus,v_m{m}*W,c) ) ),2 );
                
                E1_T0 = -T(m).*(v_m{m+1}*b.') - sum( log( 1+exp(T(m) .* bsxfun(@plus,v_m{m+1}*W,c) ) ),2 );
                E1_T1 = -T(m+1).*(v_m{m+1}*b.') - sum( log( 1+exp(T(m+1) .* bsxfun(@plus,v_m{m+1}*W,c) ) ),2 );
                
                Pswap = min(ones(size(num_data_model,1)) , exp(E0_T0 + E1_T1 - (E0_T1 + E1_T0) ) );               
                Do_swap = binornd(1,Pswap, size(Pswap,1) , size(Pswap,2));
                
                for n=1:numel(Do_swap)
                    if Do_swap(n)==1
                        vtemp = v_m{m}(n,:);
                        v_m{m}(n,:) = v_m{m+1}(n,:);
                        v_m{m+1}(n,:) = vtemp;
                        
                        htemp = h_m{m}(n,:);
                        h_m{m}(n,:) = h_m{m+1}(n,:);
                        h_m{m+1}(n,:) = htemp;
                    end
                end
                
            end
        end
        
        %% negative gradient
        v_model=mean(v_m{end},1);
        h_model=mean(h_m{end},1);
        vh_model=v_m{end}'*h_m{end}./num_data_model;
        samples = v_m{end};
        
        %% update parameters
        W=W+mu.*(vh_data-vh_model);
        b=b+mu.*(v_data-v_model);
        c=c+mu.*(h_data-h_model);
        if(max(max(isnan(W))))
            disp('Error: W Diverging');
        end
        if( max(isnan(b)) )
            disp('Error: b Diverging');
        end
        if( max(isnan(c)) )
            disp('Error: c Diverging');
        end

        %% decreasing update parameter
        mu=0.995.*mu;
       
       
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
            filename = ['rbmPT_' num2str(curr_iter) '_' today]; 
            if(printout)
                disp(['Savining File:  ' filename]);
            end
            save(filename,'samples', 'W','b','c','e','num_hidden','mu','size_batch', 'tot_iter', 'num_gibbstep', 'num_Temp', 'swap_iter');
        end
       
    end
end






end

