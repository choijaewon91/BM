function [E] = E_GRBM(v,W,b,c,var)
vW=bsxfun(@plus, bsxfun(@rdivide, v, var)*W,c);
vW_max=max(vW,0);
E = sum(bsxfun(@rdivide,  bsxfun(@minus,v,b).^2 , 2.*var),2) - sum( log( exp(-vW_max)+exp(vW-vW_max ) ) + vW_max ,2 );