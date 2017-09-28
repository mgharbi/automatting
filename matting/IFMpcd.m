% Uses the precomputed data for neighborhoods and flows

function alpha = IFMpcd(IFMdata)

    [h, w, ~] = size(IFMdata.kToU);
    N = h * w;

    CM_weights = ones(h, w);
    LOC_weights = ones(h, w);
    IU_weights = ones(h, w);
    KU_weights = ones(h, w);
    %CM_weights(:, 1 : ceil(w/2)) = 0;
    %LOC_weights(:, 1 : ceil(w/2)) = 0;
    %KU_weights(:, 1 : ceil(w/2)) = 0;
    %IU_weights(100 : 250, 450 : 600) = 0;

    cm_mult = 1;
    loc_mult = 1;
    iu_mult = 0.01;
    ku_mult = 0.05;
    lambda = 100;
    A = ...
                cm_mult * colorMixtureLaplacian(N, IFMdata.CM_inInd, IFMdata.CM_neighInd, IFMdata.CM_flows, CM_weights) + ...
                loc_mult * mattingLaplacian(N, IFMdata.LOC_inInd, IFMdata.LOC_flowRows, IFMdata.LOC_flowCols, IFMdata.LOC_flows, LOC_weights) + ...
                iu_mult * similarityLaplacian(N, IFMdata.IU_inInd, IFMdata.IU_neighInd, IFMdata.IU_flows, IU_weights) + ... 
                ku_mult * spdiags(KU_weights(:), 0, N, N) * spdiags(IFMdata.kToUconf(:), 0, N, N) + ...
                lambda * spdiags(double(IFMdata.known(:)), 0, N, N);
    
    b =  ( ...
                ku_mult * spdiags(KU_weights(:), 0, N, N) * spdiags(IFMdata.kToUconf(:), 0, N, N) + ...
                lambda * spdiags(double(IFMdata.known(:)), 0, N, N) ...
            ) * IFMdata.kToU(:);

    alpha = pcg(A, b, [], 2000);
    
    alpha(alpha < 0) = 0;
    alpha(alpha > 1) = 1;
    alpha = reshape(alpha, [h, w]);
end

function Lcm = colorMixtureLaplacian(N, inInd, neighInd, flows, weights)
    Wcm = sparse(repmat(inInd(:), [1 size(flows, 2)]), neighInd(:), flows, N, N);
    Wcm = spdiags(weights(:), 0 , N, N) * Wcm;
    Lcm = spdiags(sum(Wcm, 2), 0 , N, N) - Wcm;
    Lcm = Lcm' * Lcm;
end

function Lmat = mattingLaplacian(N, inInd, flowRows, flowCols, flows, weights)
    weights = weights(inInd);
    flows = flows .* repmat(reshape(weights, 1, 1, length(weights)) , [size(flows, 1) size(flows, 1) 1]);
    Wmat = sparse(flowRows(:), flowCols(:), flows(:), N, N);
    Wmat = (Wmat + Wmat') / 2; % Make symmetric
    Lmat = spdiags(sum(Wmat, 2), 0 , N, N) - Wmat;
end

function Lcs = similarityLaplacian(N, inInd, neighInd, flows, weights)
    weights = weights(inInd);
    flows = flows .* repmat(weights, [1 size(flows, 2)]);
    inInd = repmat(inInd, [1, size(neighInd, 2)]);
    Wcs = sparse(inInd(:), neighInd(:), flows, N, N);
    Wcs = (Wcs + Wcs') / 2; % Make symmetric
    Lcs = spdiags(sum(Wcs, 2), 0 , N, N) - Wcs;
end