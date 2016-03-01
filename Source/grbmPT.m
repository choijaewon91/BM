function [ W, b, c, e, variance, samples ] = grbmPT( visible_node, num_hidden, mu, size_batch, tot_iter, num_gibbstep, num_Temp, swap_iter, save_freq, printout, update_rate)
%% Initialize internal parameters
num_visible = size(visible_node,2);
num_data = size(visible_node,1);

W=2/(num_visible+num_hidden).*(rand(num_visible,num_hidden)-0.5);
b=zeros(1,num_visible);
c=zeros(1,num_hidden);
variance=ones(1,num_visible);
z=log(variance);

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
        vd0=bsxfun(@rdivide,v0,variance);
        h0=1./(1+exp(-bsxfun(@plus,vd0*W, c)));
        
        
        v_data = mean(vd0 ,1);
        h_data = mean(h0,1);
        vh_data = vd0'*h0./num_curr_batch;
        z_data = 1/2.*mean(bsxfun(@minus,v0,b).^2 - v0.*(h0*W'),1);
        
        
        h_r=binornd(1, h0, num_curr_batch, num_hidden);
        v_r=normrnd( bsxfun(@plus,h_r*W',b), repmat(sqrt(variance),num_curr_batch,1), num_curr_batch, num_visible);
        e(curr_iter)=mean(sum((v0-v_r).^2,2));
        if(isnan(e(curr_iter)) )
        	disp('Error: e Diverging');
    	end



        %% Parallel Tempering
        % Sampling Step
        for n = 1:num_gibbstep
            
            if (curr_iter==1 && n==1)
                for m = 1:num_Temp
                    h_m{m}=h0;
                    num_model = size(h0,1);
                    v_m{m}=v0;
                end
            end
            v_temp=cell2mat(v_m);
            m_v = mean(v_temp,1);
            var_v = var(v_temp);
            for m = 1:num_Temp
                %for first iteration initialize h_m
                
                W_pt = T(m).*W;
                c_pt = T(m).*c;
                b_pt = T(m).*b+(1-T(m)).*m_v;
                var_pt = T(m).*variance + (1-T(m)).*var_v;  
                
                
                
                hard_h=binornd(1, h_m{m}, num_model, num_hidden);
                v_m{m}=normrnd( bsxfun(@plus,hard_h*W_pt',b_pt), repmat(sqrt(var_pt),num_model,1), num_model, num_visible);
                vd_m=bsxfun(@rdivide,v_m{m},var_pt);
                h_m{m}=1./(1 + exp( bsxfun(@plus,vd_m*W_pt,c_pt) ) );
            end           
        end
        % swap every swap_iter
        if(mod(curr_iter, swap_iter)==0)
            if(printout==1)
                disp(['Fantasy Particle Swapping - Current Iteration:  ' num2str(curr_iter)]);
            end
            
            v_temp=cell2mat(v_m);
            m_v = mean(v_temp,1);
            var_v = var(v_temp);
            
            W_pt1 = T(1).*W;
            c_pt1 = T(1).*c;
            b_pt1 = T(1).*b+(1-T(1)).*m_v;
            var_pt1 = T(1).*variance + (1-T(1)).*var_v;  
            
            
            for m = 1:num_Temp-1
                
                
                
                W_pt0 = W_pt1;
                c_pt0 = c_pt1;
                b_pt0 = b_pt1;
                var_pt0 = var_pt1;  
                
                
                
                W_pt1 = T(m+1).*W;
                c_pt1 = T(m+1).*c;
                b_pt1 = T(m+1).*b+(1-T(m+1)).*m_v;
                var_pt1 = T(m+1).*variance + (1-T(m+1)).*var_v;  
                
                E0_T0 = E_GRBM(v_m{m},W_pt0,b_pt0,c_pt0,var_pt0);
                E0_T1 = E_GRBM(v_m{m},W_pt1,b_pt1,c_pt1,var_pt1);
                
                E1_T0 = E_GRBM(v_m{m+1},W_pt0,b_pt0,c_pt0,var_pt0);
                E1_T1 = E_GRBM(v_m{m+1},W_pt1,b_pt1,c_pt1,var_pt1);
                
                Pswap = min(ones(size(num_model,1)) , exp(E0_T0 + E1_T1 - (E0_T1 + E1_T0) ) );               
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
        
        %% negative gradient1
        vm0=bsxfun(@rdivide,v_m{end},variance);
        
        
        v_model = mean(vm0 ,1);
        h_model = mean(h_m{end},1);
        vh_model = vm0'*h_m{end}./num_model;
        z_model = 1/2.*mean(bsxfun(@minus,v_m{end},b).^2 - v_m{end}.*(h_m{end}*W'),1);
        
        samples = v_m{end};
        
        %% Lookahead adaptive learning rate
        if(numel(update_rate)>1)
            if(i==num_batch)
                v_mu=batch{1};
            else
                v_mu=batch{i+1};
            end
                
                E_B = E_GRBM(v_m{end},W,b,c,variance);
                if(isnan(E_B))
                    disp('E_B is NaN');
                    return;
                end
            for u=1:numel(update_rate)
                
                W_cand=W+update_rate(u).*mu.*(vh_data-vh_model);
                b_cand=b+update_rate(u).*mu.*(v_data-v_model);
                c_cand=c+update_rate(u).*mu.*(h_data-h_model);
                z_cand=z+update_rate(u).*mu.*exp(-z).*(z_data-z_model);
                var_cand = exp(max(log(1e-9) ,min(z_cand,log(50) )));
                


                E_D = E_GRBM(v_mu,W_cand,b_cand,c_cand,var_cand);
                if(isnan(E_D))
                    disp('E_D is NaN');
                    return;
                end
                E_M = E_GRBM(v_m{end},W_cand,b_cand,c_cand,var_cand);
                if(isnan(E_M))
                    disp('E_M is NaN');
                    return;
                end
                cost(u)=sum(-E_D-logsum(E_B-E_M) + log (size(v_mu,1)) );
                if(printout)
                    if(isnan(cost(u)))
                        disp('Cost is NaN');
                        return;
                    end
                end
            end
        end
        [dump, ind] = max(cost);
        mu=update_rate(ind).*mu;
        
       	%% update parameters
       	W=W+mu.*(vh_data-vh_model);
       	b=b+mu.*(v_data-v_model);
       	c=c+mu.*(h_data-h_model);
       	z=z+mu.*exp(-z).*(z_data-z_model);
        z=max(log(1e-9),min(z,log(50)));
       	variance = exp(z);

       	if(max(max(isnan(W))))
        	disp('Error: W Diverging');
            return;
    	end
		if( max(isnan(b)) )
        	disp('Error: b Diverging');
            return;
    	end
    	if( max(isnan(c)) )
        	disp('Error: c Diverging');
            return;
    	end
    	if( max(isnan(z)) )
        	disp('Error: z Diverging');
    	end




       %% decreasing update parameter
       %mu=0.995.*mu;

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
            filename = ['grbmPT_' num2str(curr_iter) '_' today]; 
            if(printout)
                disp(['Savining File:  ' filename]);
            end
            save(filename,'variance','samples', 'W','b','c','e','num_hidden','mu','size_batch', 'tot_iter', 'num_gibbstep', 'num_Temp', 'swap_iter');
        end
       
    end
end






end

