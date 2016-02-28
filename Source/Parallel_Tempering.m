function [ output_args ] = Parallel_Tempering( v, h, i, W, b, c, T)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    for n = 1:num_gibbstep
        for m = 1:num_Temp
            %for first iteration initialize h_m
            if (i==1 && j==1 && k==1)
                h_m{m}=h0;
            end
            hard_h=binornd(1, h_m{m}, num_curr_batch, num_hidden);
            v_m{m}=1./(1 + exp( -T(m) .* bsxfun(@plus,hard_h*W',b) ) );
            h_m{m}=1./(1 + exp( -T(m) .* bsxfun(@plus,v_m{m}*W,c) ) );
        end           
    end
    % swap every swap_iter
    if(mod(i, swap_iter)==0)
        for m = 1:num_Temp-1
            E0_T0 = -T(m).*(v_m{m}*b.') - sum( log( 1+exp(T(m) .* bsxfun(@plus,v_m{m}*W,c) ) ),2 );
            E0_T1 = -T(m+1).*(v_m{m}*b.') - sum( log( 1+exp(T(m+1) .* bsxfun(@plus,v_m{m}*W,c) ) ),2 );

            E1_T0 = -T(m).*(v_m{m+1}*b.') - sum( log( 1+exp(T(m) .* bsxfun(@plus,v_m{m+1}*W,c) ) ),2 );
            E1_T1 = -T(m+1).*(v_m{m+1}*b.') - sum( log( 1+exp(T(m+1) .* bsxfun(@plus,v_m{m+1}*W,c) ) ),2 );

            Pswap = min(ones(size(E0_T0)) , exp(E0_T0 + E1_T1 - (E0_T1 + E1_T0) ) );               
            Do_swap = binornd(1,Pswap, size(Pswap,1) , size(Pswap,2));

            for n=1:numel(Do_swap)
                if Doswap(n)==1
                    vtemp = v{m}(n,:);
                    v{m}(n,:) = v{m+1}(n,:);
                    v{m+1}(n,:) = vtemp;

                    htemp = h{m}(n,:);
                    h{m}(n,:) = h{m+1}(n,:);
                    h{m+1}(n,:) = htemp;
                end
            end

        end
    end

end

