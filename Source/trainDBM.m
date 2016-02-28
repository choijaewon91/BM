function [ W, b ,c, e,Pv] = trainDBM( v, mu, num_h, n, lrnRate,batch)
    num_v=size(v,2);
    ndata=size(v,1);
    lrCand=numel(lrnRate);
    
    layer=numel(num_h);
    vd=v;
    
    
    
    
    
    c=cell(layer,1);
    for i=1:layer
        c{i}=zeros(1,num_h(i));
    end
    
    
    
    b=zeros(1,num_v);
    
    
    W=cell(layer,1);
    for i=1:layer
        if i==1
            W{i}=2./(num_v+num_h(1)).*rand(num_v,num_h(1))-1./(num_v+num_h(1));
        else
            W{i}=2./(num_h(i-1)+num_h(i)).*rand(num_h(i-1),num_h(i))-1./(num_h(i-1)+num_h(i));
        end
    end
    
    
    
    
    for i=1:n
        idx=randi([1 ndata],[1,batch]);
        vb=vd(idx,:);
        c_mat=cell(layer,1);
        for j=1:layer
            c_mat{j}=repmat(c{j},batch,1);
        end
        b_mat=repmat(b,batch,1);    
        
        
        %<vh>_data
        % mean-field approx.
        h=cell(layer,1);
        Ph=cell(layer,1);
        for j=1:layer
            h{j}=rand(batch,num_h(j));
            Ph{j}=rand(batch,num_h(j));
        end
        iter=0;
        while 1
            for j=1:layer-1
                if j==1
                    Ph{j}=1./( 1+exp(-(c_mat{j} + (vb)*W{1} + Ph{2}*(W{2})' ) ) );
                else
                    Ph{j}=1./( 1+exp(-(c_mat{j} + Ph{j-1}*W{j} + Ph{j+1}*W{j+1}' ) ) );
                end
            end
            Ph{layer}=1./( 1+exp(-(c_mat{layer} + Ph{layer-1}*W{layer}) ) );

            h_sum=1;
            if iter~=0
                h_sum=0;
                for k=1:layer
                    h_sum=sum(sum(abs(old_Ph{k}-Ph{k})))+h_sum;
                end
            end
            if h_sum < 1e-7
                break;
            end
            old_Ph=Ph;
            iter=iter+1;
        end
        for j=1:layer
            h{j}=ceil(Ph{j}-rand(batch,num_h(j)));
        end
        %first gradient term
        W_data=cell(layer,1);
        c_data=cell(layer,1);
        for j=1:layer
            if j==1
                W_data{j}=vb'*h{j}./batch;
            else
                W_data{j}=h{j-1}'*h{j}./batch;
            end
            
            c_data{j}=sum(h{j})./batch;
            
        end
        b_data=sum(vb,1)./batch;
        
        
        
        %<vh>_model
        if i==1
            h_r=h;
            vo=vb;
            PrePv=vo;
            Ph_r=Ph;
        end
        % down to v
        Preh_r=h_r;
        Pv=1./(1+exp(-(b_mat +Preh_r{1}*W{1}')));
        
        vr=Pv;
        %vr=ceil(Pv-rand(batch,num_v));
        % up to h
        
        for j=1:layer-1
            if j==1
                Ph_r{j}=1./( 1+exp(-(c_mat{j} + (PrePv)*W{1} + Preh_r{2}*W{2}' ) ) );
                h_r{j}=ceil(Ph_r{j}-rand(batch,num_h(j)));
            else
                Ph_r{j}=1./( 1+exp(-(c_mat{j} + Preh_r{j-1}*W{j} + Preh_r{j+1}*W{j+1}' ) ) );
                h_r{j}=ceil(Ph_r{j}-rand(batch,num_h(j)));
            end
        end
        Ph_r{layer}=1./( 1+exp(-(c_mat{layer} + Preh_r{layer-1}*W{layer}) ) );
        h_r{layer}=ceil(Ph_r{layer}-rand(batch,num_h(layer)));
        PrePv=Pv;
        
        
        
        %Second gradient term
        W_recon=cell(layer,1);
        c_recon=cell(layer,1);
        for j=1:layer
            if j==1
                W_recon{j}=vr'*h_r{j}./batch;
            else
                W_recon{j}=h_r{j-1}'*h_r{j}./batch;
            end
            
            c_recon{j}=sum(h_r{j})./batch;
            
           
        end
        b_recon=sum(vr,1)./batch;
        
        
%         Pvd=zeros(1,lrCand);
%         for j=1:lrCand
%             Wcand=W+ lrnRate(j).*mu.*(vh_data-vh_recon);
%             bcand=b+ lrnRate(j).*mu.*(v_data-v_recon);
%             ccand=c+ lrnRate(j).*mu.*(h_data-h_recon);
%             bc_mat=repmat(bcand,batch,1);
%             cc_mat=repmat(ccand,batch,1);            
% 
%             %calculate h candidates
%             Ph_cand=1./( 1+exp(-(cc_mat + vb*Wcand) ) );
%             %h_cand=Ph_cand;
%             
%             %calculate P(v|W,b,c,var)
%             mu_g=bc_mat+Ph_cand*W';
%             
%             Pvd(j)=sum( prod(abs(1-vb-1./(1+exp(-mu_g))) ))./batch;
%         end
%         [M, I]=max(Pvd);
         disp(i);
%         mu=mu.*lrnRate(I);
%         if mu>maxmu
%             mu=maxmu;
%         elseif mu< minmu
%             mu=minmu;
%         end
        for j=1:layer
            W{j}=W{j}+mu.*(W_data{j}-W_recon{j});
            c{j}=c{j}+mu.*(c_data{j}-c_recon{j});
        end
        
        b=b+mu.*(b_data-b_recon);
        
                
        e(i)=sum(sum((vo-Pv).^2)./batch)./num_v;
    end
    figure(1)
    for i=1:100
        subplot(10,10,i)
        imagesc(reshape(Pv(i,:),28,28))
        colormap(gray)
        set(gca,'YTickLabel', [],'XTickLabel', []);
    end
end

