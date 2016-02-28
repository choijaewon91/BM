function [ W, b ,c, e,variance] = trainGDBM( v, mu, num_h_list, n, recur, lrnRate, maxmu, minmu )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    depth=numel(num_h_list);
    num_v=size(v,2);
    ndata=size(v,1);
    lrCand=numel(lrnRate);
    
    %initializing std dev(z), b, c, W 
    W=cell(1,depth);
    c=cell(1,depth);
    b=cell(1,depth);
    
    num_h=num_h_list(1);
    vd=v;    
    variance=ones(1,num_v);
    z=log(variance);
    c{1}=zeros(1,num_h);
    b{1}=zeros(1,num_v);
    W{1}=2./(num_v+num_h).*rand(num_v,num_h)-1./(num_v+num_h);
    
    
    %GRBM training
    for i=1:n        
        c_mat=repmat(c{1},ndata,1);
        b_mat=repmat(b{1},ndata,1);
        var_v=repmat(variance,ndata,1);
        
        %<vh>_data
            %calculate hidden layer
        Ph=1./( 1+exp(-(c_mat + (vd./var_v)*W{1}) ) );
        h=ceil(Ph-rand(ndata,num_h(1)));
        % derivative of energy function of data given parameters z, b, c, W
        vh_data=(vd./var_v)'*h./ndata;
        v_data=sum(vd./var_v,1)./ndata;
        h_data=sum(h,1)./ndata;
        z_data=sum(1/2.*(vd-b_mat).^2-vd.*(h*W{1}'))./ndata;
        
        
        %<vh>_recon
            %Gibbs Step Iteration
        for k=1:recur
            mu_r=b_mat+h*W';
            Pv=normrnd(mu_r,sqrt(var_v),ndata,num_v);
        
            Ph_r=1./(1+exp(-(c_mat +(Pv./var_v)*W)));
            h=ceil(Ph-rand(ndata,num_h));
        end
        
        % derivative of energy function of model given parameters z, b, c, W
        vh_recon=(Pv./var_v)'*Ph_r./ndata;
        v_recon=sum(Pv./var_v)./ndata;
        h_recon=sum(Ph_r)./ndata;
        z_recon=sum(1/2.*(Pv-b_mat).^2-Pv.*(Ph_r*W'))./ndata;
        
        
        % look ahead adaptive learning rate
        Pvd=zeros(1,lrCand);
        for j=1:lrCand
            
            %calculate parameters for each candidate learning rates per
            %cycle
            Wcand=W+ lrnRate(j).*mu.*(vh_data-vh_recon);
            bcand=b+ lrnRate(j).*mu.*(v_data-v_recon);
            ccand=c+ lrnRate(j).*mu.*(h_data-h_recon);
            bc_mat=repmat(bcand,ndata,1);
            cc_mat=repmat(ccand,ndata,1);
            zcand=z+ lrnRate(j).*mu.*(exp(-z)).*(z_data-z_recon);
            varcand=exp(zcand);
            varc_mat=repmat(varcand,ndata,1);
            sigmat=diag(varcand);
            
            %calculate h candidates
            Ph_cand=1./( 1+exp(-(cc_mat + (vd./varc_mat)*Wcand) ) );
            h_cand=ceil(Ph_cand-rand(ndata,num_h));
            %h_cand=Ph_cand;
            
            %calculate P(v|W,b,c,var)
            mu_g=bc_mat+h_cand*W';
            Pvd(j)=sum(mvnpdf(vd,mu_g,sigmat))./ndata;
        end
        % find learning rate with maximum P(v|W,b,c,var)
        [M, I]=max(Pvd);
 
        % calculate new learning rate (ceil and floor if too high or low) 
        mu=mu.*lrnRate(I);
        if mu>maxmu
            mu=maxmu;
        elseif mu< minmu
            mu=minmu;
        end
        
        % new parameters
        W{1}=W{1}+mu.*(vh_data-vh_recon);
        b{1}=b{1}+mu.*(v_data-v_recon);
        c{1}=c{1}+mu.*(h_data-h_recon);
        z=z +mu.*(exp(-z)).*(z_data-z_recon);
        variance=exp(z);
        
        e(i)=sum(sum((vd-Pv).^2)./ndata)./num_v;
    end
    
    
    % DBM Training
    
    Ph=1./( 1+exp(-(c_mat + (vd./var_v)*W{1}) ) );
    h=ceil(Ph-rand(ndata,num_h(1)));
    
    
    
    for k=2:depth
        clear(vd);
        vd=h;
        num_v=num_h_list(k-1);
        num_h=num_h_list(k);
        c{k}=zeros(1,num_h);
        b{k}=zeros(1,num_v);
        W{k}=2./(num_v+num_h).*rand(num_v,num_h)-1./(num_v+num_h);
        
        c_mat=repmat(c{k},ndata,1);
        b_mat=repmat(b{k},ndata,1);
        
        for i=1:n
            %<vh>_data
            %calculate hidden layer
            Ph=1./( 1+exp(-(c_mat + (vd./var_v)*W{1}) ) );
            h=ceil(Ph-rand(ndata,num_h(1)));
            % derivative of energy function of data given parameters z, b, c, W
            vh_data=(vd./var_v)'*h./ndata;
            v_data=sum(vd./var_v,1)./ndata;
            h_data=sum(h,1)./ndata;
            z_data=sum(1/2.*(vd-b_mat).^2-vd.*(h*W{1}'))./ndata;
        end
    end
end
