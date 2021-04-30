function x = lineSearch_reloaded(x,objFunc,options)
% 2010 m.bangert@dkfz.de
% mode       = 'fletcher'; % 'fletcher', 'hestenes' or 'polak'
% linesearch = 'exact'; % 'armijo', 'quadratic' or 'exact'

mode = options.mode;
linesearch= options.linesearch;

[objFuncValue,dx]    = objFunc(x); % returns also the gradients
oldObjFuncValue = objFuncValue + 1;

% implementation according to nocedal 5.2 algorithm 5.4 (FR-CG)
% polak and hestenes updates for beta can be found in the same section
%dx  = gradFunc(x);
dir = -dx;

% iterate
iter      = 0;
numOfIter = options.numIterations;
prec      = 1e-5;
maxStepLength = options.maxStepLength;

% convergence if gradient smaller than prec, change in objective function
% smaller than prec or maximum number of iteration reached...
while iter < numOfIter && abs((oldObjFuncValue-objFuncValue)/objFuncValue)>prec && norm(dx)>prec
    
    % iteration counter
    iter = iter + 1;
    
    if strcmp(linesearch,'armijo')
        alpha = mb_backtrackingLineSearch(objFunc,objFuncValue,x,dx,dir);
    elseif strcmp(linesearch,'exact')
        linesearchFunc = @(delta) objFunc(x+delta*dir);
        alpha          = fminbnd(linesearchFunc,0,maxStepLength); % matlab inherent function to find 1d min
    elseif strcmp(linesearch,'quadratic')
        alpha = mb_quadraticApproximationLineSearch(objFunc,objFuncValue,x,dx,dir);
    end
        
    % update x
    x = x + alpha*dir;

    % update obj func values and update dx
    oldDx = dx;
    oldObjFuncValue = objFuncValue;
    [objFuncValue,dx]    = objFunc(x);
    
    %dx    = gradFunc(x);
        
    if strcmp(mode,'fletcher')
        beta = (dx'*dx)/(oldDx'*oldDx);
    elseif strcmp(mode,'hestenes')
        beta = (dx'*(dx-oldDx))/((dx-oldDx)'*dir);
    elseif strcmp(mode,'polak')
        beta = (dx'*(dx-oldDx))/(oldDx'*oldDx);
    end
    
    % update search direction
    dir = -dx + beta*dir;
           
    if options.verbosity == 1
        fprintf(['Iteration ' num2str(iter) ' - Obj Func = ' num2str(objFuncValue) ' @ x = [' num2str(x') ']\n']);
    end
    
end
