function [err, munew2, sigmanew2, kpb] = ...
    check_posdef(munew, sigmanew, muold, sigmaold)

sigmanew2 = sigmanew;
munew2 = munew;
K = size(munew,1);
kpb = [];

for k=1:K
    C = squeeze(sigmanew(k,:,:));
    %s = sqrt(diag(C));
    %if (any(s~=1)), C = C ./ (s * s');
    %end
    [~,err] = cholcov(C,0);
    if err ~= 0 
        if 0
            sigmanew2(k,:,:) = sigmaold(k,:,:);
        else
            %sigmanew2(k,:,:) = nearestSPD(C);
            [sigmanew2(k,:,:), ~, flagout] = nearcorr(C);            
            if not(flagout), sigmanew2(k,:,:) = nearestSPD(C); end
        end
        %munew2(:,k) = muold(:,k);
        kpb = [kpb,k]; 
        break;
    end
end
