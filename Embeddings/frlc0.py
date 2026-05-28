# Code du papier :
import torch
import matplotlib.pyplot as plt


def compute_grad_A(C, Q, R, Lambda, gamma, \
                   semiRelaxedLeft, semiRelaxedRight, device, \
                   Wasserstein=True, FGW=False, A=None, B=None, \
                   alpha=0.0, unbalanced=False, \
                   dtype=torch.float64, full_grad=True):
    
    '''
    Code for computing the Wasserstein, Gromov-Wasserstein, and fused Gromov-Wasserstein gradients with respect to Q and R.
    These depend on the marginal relaxation of the specific OT problem you want to solve, due to proportionality simplifications.
    
    ------Parameters------
    C: torch.tensor (N1 x N2)
        A matrix of pairwise feature distances in space X and space Y (inter-space).
    Q: torch.tensor (N1 x r)
        The left sub-coupling matrix.
    R: torch.tensor (N2 x r)
        The right sub-coupling matrix.
    Lambda: torch.tensor (r x r)
        The inner transition matrix.
    gamma: float
        The mirror-descent step-size.
    semiRelaxedLeft: bool
        True if relaxing the left marginal.
    semiRelaxedRight: bool
        True if relaxing the right marginal.
    device: str
        The device (i.e. 'cpu' or 'cuda')
    Wasserstein: bool
        True if using the Wasserstein loss <C, P>_F as the objective cost,
        else runs GW if FGW false and FGW if GW true.
    FGW: bool
        True if running the Fused-Gromov Wasserstein problem, and otherwise false.
    A: torch.tensor (N1 x N1)
        Pairwise distance matrix in metric space X.
    B: torch.tensor (N2 x N2)
        Pairwise distance matrix in metric space Y.
    alpha: float
        A balance parameter between the Wasserstein term and
        the Gromov-Wasserstein term of the objective.
    unbalanced: bool
        True if running the unbalanced problem;
        if semiRelaxedLeft/Right and unbalanced False (default) then running the balanced problem.
    '''
    
    r = Lambda.shape[0]
    one_r = torch.ones((r), device=device, dtype=dtype)
    One_rr = torch.outer(one_r, one_r).to(device)
    
    if Wasserstein:
        gradQ, gradR = Wasserstein_Grad(C, Q, R, Lambda, device, \
                   dtype=torch.float64, full_grad=full_grad)
        
    elif A is not None and B is not None:
        if not semiRelaxedLeft and not semiRelaxedRight and not unbalanced:
            # Balanced gradient (Q1_r = a AND R1_r = b)
            gradQ = - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = - 4 * (B@R@Lambda.T)@(Q.T@A@Q)@Lambda
        elif semiRelaxedRight:
            # Semi-relaxed right marginal gradient (Q1_r = a)
            gradQ = - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = 2*B**2 @ R @ One_rr - 4*(B@R@Lambda.T)@(Q.T@A@Q)@Lambda
        elif semiRelaxedLeft:
            # Semi-relaxed left marginal gradient (R1_r = b)
            gradQ = 2*A**2 @ Q @ One_rr - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = - 4 * (B@R@Lambda.T)@(Q.T@A@Q)@Lambda
        elif unbalanced:
            # Fully unbalanced with no marginal constraints
            gradQ = 2*A**2 @ Q @ One_rr - 4 * (A@Q)@Lambda@(R.T@B@R)@Lambda.T
            gradR = 2*B**2 @ R @ One_rr - 4 * (B@R@Lambda.T)@(Q.T@A@Q)@Lambda

        if full_grad:
            N1, N2 = Q.shape[0], R.shape[0]
            one_N1, one_N2 = torch.ones((N1), device=device, dtype=dtype), torch.ones((N2), device=device, dtype=dtype)
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            F = (Q@Lambda@R.T)
            MR = Lambda.T @ Q.T @ A @ F @ B @ R @ torch.diag(1/gR)
            MQ = Lambda @ R.T @ B @ F.T @ A @ Q @ torch.diag(1/gQ)
            gradQ += 4*torch.outer(one_N1, torch.diag(MQ))
            gradR += 4*torch.outer(one_N2, torch.diag(MR))
        
        # Readjust cost for FGW problem
        if FGW:
            gradQW, gradRW = Wasserstein_Grad(C, Q, R, Lambda, device,\
                   dtype=torch.float64, full_grad=full_grad)
            gradQ = (1-alpha)*gradQW + alpha*gradQ
            gradR = (1-alpha)*gradRW + alpha*gradR
    else:
        raise ValueError("---Input either Wasserstein=True or provide distance matrices A and B for GW problem---")
        
    normalizer = torch.max(torch.tensor([torch.max(torch.abs(gradQ)) , torch.max(torch.abs(gradR))]))
    gamma_k = gamma / normalizer
    
    return gradQ, gradR, gamma_k


def compute_grad_B(C, Q, R, Lambda, gQ, gR, gamma, device, Wasserstein=True, \
                   FGW=False, A=None, B=None, alpha=0.0, \
                   dtype=torch.float64):
    '''
    Code for computing the Wasserstein, Gromov-Wasserstein, and fused Gromov-Wasserstein gradients with respect to T.
    
    ------Parameters------
    C: torch.tensor (N1 x N2)
        A matrix of pairwise feature distances in space X and space Y (inter-space).
    Q: torch.tensor (N1 x r)
        The left sub-coupling matrix.
    R: torch.tensor (N2 x r)
        The right sub-coupling matrix.
    Lambda: torch.tensor (r x r)
        The inner transition matrix.
    gQ: torch.tensor (r)
        The inner marginal corresponding to the matrix Q.
    gR: torch.tensor (r)
        The inner marginal corresponding to the matrix R.
    gamma: float
        The mirror-descent step-size.
    device: str
        The device (i.e. 'cpu' or 'cuda')
    Wasserstein: bool
        True if using the Wasserstein loss <C, P>_F as the objective cost,
        else runs GW if FGW false and FGW if GW true.
    FGW: bool
        True if running the Fused-Gromov Wasserstein problem, and otherwise false.
    A: torch.tensor (N1 x N1)
        Pairwise distance matrix in metric space X.
    B: torch.tensor (N2 x N2)
        Pairwise distance matrix in metric space Y.
    alpha: float
        A balance parameter between the Wasserstein term and
        the Gromov-Wasserstein term of the objective.
    '''
    if Wasserstein:
        gradLambda = Q.T @ C @ R
    else:
        gradLambda = -4 * Q.T @ A @ Q @ Lambda @ R.T @ B @ R
        if FGW:
            gradLambda = (1-alpha)*(Q.T @ C @ R) + alpha*gradLambda
    gradT = torch.diag(1/gQ) @ gradLambda @ torch.diag(1/gR) # (mass-reweighted form)
    gamma_T = gamma / torch.max(torch.abs(gradT))
    return gradT, gamma_T


def Wasserstein_Grad(C, Q, R, Lambda, device, \
                   dtype=torch.float64, full_grad=True):
    
    gradQ = (C @ R) @ Lambda.T
    if full_grad:
        # rank-one perturbation
        N1 = Q.shape[0]
        one_N1 = torch.ones((N1), device=device, dtype=dtype)
        gQ = Q.T @ one_N1
        w1 = torch.diag( (gradQ.T @ Q) @ torch.diag(1/gQ) )
        gradQ -= torch.outer(one_N1, w1)
    
    # linear term
    gradR = (C.T @ Q) @ Lambda
    if full_grad:
        # rank-one perturbation
        N2 = R.shape[0]
        one_N2 = torch.ones((N2), device=device, dtype=dtype)
        gR = R.T @ one_N2
        w2 = torch.diag( torch.diag(1/gR) @ (R.T @ gradR) )
        gradR -= torch.outer(one_N2, w2)
    
    return gradQ, gradR


def FRLC_opt(C,
             a=None,
             b=None,
             A=None,
             B=None,
             tau_in=50,
             tau_out=50,
             gamma=70, 
             r = 40, 
             r2=None, 
             max_iter=200, 
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
             dtype=torch.float64, 
             semiRelaxedLeft=False, 
             semiRelaxedRight=False, 
             Wasserstein=True, 
             printCost=True, 
             returnFull=True, 
             FGW=False, 
             alpha=0.0, 
             unbalanced=False, 
             initialization='Full',
             init_args = None,
             full_grad=True,
             convergence_criterion=True,
             tol=1e-5,
             min_iter = 25,
             min_iterGW = 500,
             max_iterGW = 1000,
             max_inneriters_balanced= 500,
             max_inneriters_relaxed= 500):
    
    '''
    FRLC: Factor Relaxation with Latent Coupling
    ------Parameters------
    C: torch.tensor (N1 x N2)
        A matrix of pairwise feature distances in space X and space Y (inter-space).
    a: torch.tensor (N1)
        A vector representing marginal one.
    b: torch.tensor (N2)
        A vector representing marginal two.
    A: torch.tensor (N1 x N1)
        A matrix of pairwise distances between points in metric space X.
    B: torch.tensor (N2 x N2)
        A matrix of pairwise distances between points in metric space Y.
    tau_in: float (> 0)
        A scalar which controls the regularity of the inner marginal update path.
    tau_out: float (> 0)
        A scalar which controls the regularization of the outer marginals a and b (e.g. for semi-relaxed or unbalanced OT)
    gamma: float (> 0)
        The mirror descent step-size, a scalar which controls the scaling of gradients
        before being exponentiated into Sinkhorn kernels.
    r: int (> 1)
        A non-negative integer rank, controlling the rank of the FRLC learned OT coupling. 
    max_iter: int
        The maximal number of iterations FRLC will run until convergence.
    device: str
        The device (i.e. 'cpu' or 'cuda') which FRLC runs on.
    dtype: dtype
        The datatype all tensors are stored on (naturally there is a space-accuracy
        tradeoff for low-rank between 32 and 64 bit).
    semiRelaxedLeft: bool
        True if running the left-marginal relaxed low-rank algorithm.
    semiRelaxedRight: bool
        True if running the right-marginal relaxed low-rank algorithm.
    Wasserstein: bool
        True if using the Wasserstein loss <C, P>_F as the objective cost,
        else runs GW if FGW false and FGW if GW true.
    printCost: bool
        True if printing the value of the objective cost at each iteration.
        This can be expensive for large datasets if C is not factored.
    returnFull: bool
        True if returning P_r = Q Lambda R.T, else returns iterates (Q, R, T).
    FGW: bool
        True if running the Fused-Gromov Wasserstein problem, and otherwise false.
    alpha: float
        A balance parameter between the Wasserstein term and
        the Gromov-Wasserstein term of the objective.
    unbalanced: bool
        True if running the unbalanced problem;
        if semiRelaxedLeft/Right and unbalanced False (default) then running the balanced problem.
    initialization: str, 'Full' or 'Rank-2'
        'Full' if sub-couplings initialized to be full-rank, if 'Rank-2' set to a rank-2 initialization.
        We advise setting this to be 'Full'.
    init_args: tuple of 3-tensors
        A tuple of (Q0, R0, T0) for tuple[i] of type tensor
    full_grad: bool
        If True, evaluates gradient with rank-1 perturbations.
        Else if False, omits perturbation terms.
    convergence_criterion: bool
        If True, use the convergence criterion. Else if False, default to running up to max_iters.
    tol: float
        Tolerance used for established when convergence is reached.
    min_iter: int
        The minimum iterations for the algorithm to run for in the Wasserstein case.
    min_iterGW: int
        The minimum number of iterations to run for in the GW case.
    max_iterGW: int
        The maximum number of iterations to run for in the GW case.
    max_inneriters_balanced: int
        The maximum number of inner iterations for the Sinkhorn loop.
    max_inneriters_relaxed: int
        The maximum number of inner iterations for the relaxed and semi-relaxed loops.
    '''
    N1, N2 = C.size(dim=0), C.size(dim=1)
    k = 0
    
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    
    if a is None:
        a = one_N1 / N1
    if b is None:
        b = one_N2 / N2
    if r2 is None:
        r2 = r

    one_r = torch.ones((r), device=device, dtype=dtype)
    one_r2 = torch.ones((r2), device=device, dtype=dtype)
    
    # Initialize inner marginals to uniform; 
    # generalized to be of differing dimensions to account for non-square latent-coupling.
    gQ = (1/r)*one_r
    gR = (1/r2)*one_r2
    
    full_rank = True if initialization == 'Full' else False
    
    if initialization == 'Full':
        full_rank = True
    elif initialization == 'Rank-2':
        full_rank = False
    else:
        full_rank = True
        print('Initialization must be either "Full" or "Rank-2", defaulting to "Full".')
        
    if init_args is None:
        Q, R, T, Lambda = initialize_couplings(a, b, gQ, gR, gamma, full_rank=full_rank,\
                                                device=device, dtype=dtype,\
                                                max_iter = max_inneriters_balanced)
        
        Q, R, T = stabilize_Q_init(Q, device=device, dtype=dtype), stabilize_Q_init(R, device=device, dtype=dtype), stabilize_Q_init(T, device=device, dtype=dtype)
        
        Lambda = torch.diag(1/ (Q.T @ one_N1)) @ T @ torch.diag(1/ (R.T @ one_N2))
    else:
        Q, R, T = init_args
        Lambda = torch.diag(1/ (Q.T @ one_N1)) @ T @ torch.diag(1/ (R.T @ one_N2))

    if Wasserstein is False:
        min_iter = min_iterGW
        max_iter = max_iterGW

    '''
    Preparing main loop.
    '''
    errs = []
    gamma_k = gamma
    
    Q_prev, R_prev, T_prev = None, None, None
    
    # Initialize duals for warm-start across iterations
    dual_1Q, dual_2Q = None, None
    dual_1R, dual_2R = None, None
    dual_1T, dual_2T = None, None
    
    while (k < max_iter and (not convergence_criterion or \
                       (k < min_iter or Delta((Q, R, T), (Q_prev, R_prev, T_prev), gamma_k) > tol))):
        
        if convergence_criterion:
            # Set previous iterates to evaluate convergence at the next round
            Q_prev, R_prev, T_prev = Q, R, T
        
        if k % 25 == 0:
            print(f'Iteration: {k}')
            print(f'Erreur: {errs}')
        
        gradQ, gradR, gamma_k = compute_grad_A(C, Q, R, Lambda, gamma, semiRelaxedLeft, \
                                               semiRelaxedRight, device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha, \
                                                  unbalanced=unbalanced, full_grad=full_grad)
        if semiRelaxedLeft:
            
            R, dual_1R, dual_2R = logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in, dual_1 = dual_1R, dual_2 = dual_2R)
            Q, dual_1Q, dual_2Q = logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in, \
                                 dual_1 = dual_1Q, dual_2 = dual_2Q)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            
            gradT, gamma_T = compute_grad_B(C, Q, R, Lambda, gQ, gR, \
                                               gamma, device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha)
            
        elif semiRelaxedRight:
            
            Q, dual_1Q, dual_2Q = logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in, \
                                 dual_1 = dual_1Q, dual_2 = dual_2Q)
            R, dual_1R, dual_2R = logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in, \
                                 dual_1 = dual_1R, dual_2 = dual_2R)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            
            gradT, gamma_T = compute_grad_B(C, Q, R, Lambda, gQ, gR, \
                                               gamma, device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha)
            
        elif unbalanced:
            
            Q, dual_1Q, dual_2Q = logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in, \
                                dual_1 = dual_1Q, dual_2 = dual_2Q)
            
            R, dual_1R, dual_2R = logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in, \
                                dual_1 = dual_1R, dual_2 = dual_2R)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            
            gradT, gamma_T = compute_grad_B(C, Q, R, Lambda, gQ, gR, gamma, \
                                               device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha)
            
        else:
            
            # Balanced
            Q, dual_1Q, dual_2Q = logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in, \
                                dual_1 = dual_1Q, dual_2 = dual_2Q)
            
            R, dual_1R, dual_2R = logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                         device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in, \
                                dual_1 = dual_1R, dual_2 = dual_2R)
            
            gQ, gR = Q.T @ one_N1, R.T @ one_N2
            
            gradT, gamma_T = compute_grad_B(C, Q, R, Lambda, gQ, gR, gamma, \
                                               device, Wasserstein=Wasserstein, \
                                               A=A, B=B, FGW=FGW, alpha=alpha)
        
        T, dual_1T, dual_2T = logSinkhorn(gradT - (gamma_T**-1)*torch.log(T), gQ, gR, gamma_T, max_iter = max_inneriters_balanced, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False, \
                                dual_1 = dual_1T, dual_2 = dual_2T)
        
        # Inner latent transition-inverse matrix
        Lambda = torch.diag(1/gQ) @ T @ torch.diag(1/gR)
        
        if printCost:
            if Wasserstein:
                cost = torch.trace(((Q.T @ C) @ R) @ Lambda.T)
            else:
                P = Q @ Lambda @ R.T
                M1 = Q.T @ A**2 @ Q
                M2 = R.T @ B**2 @ R
                cost = one_r.T @ M1 @ one_r + one_r.T @ M2 @ one_r -2*torch.trace((A @ P @ B).T @ P)
                if FGW:
                    cost = (1-alpha)*torch.sum(C * P) + alpha*cost
            errs.append(cost.cpu())
            
        k+=1
    if printCost:
        ''' 
        Plotting OT objective value across iterations.
        '''
        plt.plot(range(len(errs)), errs)
        plt.xlabel('Iterations')
        plt.ylabel('OT-Cost')
        plt.show()
        '''
        Plotting latent coupling.
        '''
        plt.imshow(T.cpu())
        plt.show()
    
    if returnFull:
        P = Q @ Lambda @ R.T
        return P, errs


def initialize_couplings(a, b, gQ, gR, gamma, \
                         full_rank=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), \
                         dtype=torch.float64, rank2_random=False, \
                        max_iter=50):
    '''
    ------Parameters------
    a: torch tensor
        Left outer marginal, should be positive and sum to 1.0
    b: torch tensor
        Right outer marginal, should be positive and sum to 1.0
    gQ: torch tensor
        Left inner marginal, should be positive and sum to 1.0
    gR: torch tensor
        Right inner marginal, should be positive and sum to 1.0
    gamma: float
        Step-size of the coordinate MD
    full_rank: bool
        If True, initialize a full-rank set of sub-couplings.
        Else if False, initialize with a rank-2 initialization.
    device: str
        'cpu' if running on CPU, else 'cuda' for GPU
    dtype: torch dtype
        Defaults to float64
    rank2_random: bool
        If False, use deterministic rank 2 initialization of Scetbon '21
        Else, use an initialization with randomly sampled vector on simplex.
    max_iter: int
        The maximum number of Sinkhorn iterations for initialized sub-couplings.
    '''
    N1, N2 = a.size(dim=0), b.size(dim=0)
    r, r2 = gQ.size(dim=0), gR.size(dim=0)
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    
    if full_rank:
        '''
        A means of initializing full-rank sub-coupling matrices using randomly sampled matrices
        and Sinkhorn projection onto the polytope of feasible couplings.

        Only non-diagonal initialization for the LC-factorization and handles the case of unequal
        inner left and right ranks (non-square latent couplings).
        '''
        # 1. Q-generation
        # Generate a random (full-rank) matrix as our coupling initialization
        C_random = torch.rand((N1,r), device=device, dtype=dtype)
        Q,_,_ = logSinkhorn(C_random, a, gQ, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # 2. R-generation
        C_random = torch.rand((N2,r2), device=device, dtype=dtype)
        '''
        xi_random = torch.exp( -C_random )
        u, v = Sinkhorn(xi_random, b, gR, N2, r2, gamma, device=device, max_iter=max_iter, dtype=dtype)
        R = torch.diag(u) @ xi_random @ torch.diag(v)'''
        R,_,_ = logSinkhorn(C_random, b, gR, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # 3. T-generation
        gR, gQ = R.T @ one_N2, Q.T @ one_N1
        C_random = torch.rand((r,r2), device=device, dtype=dtype)
        '''
        xi_random = torch.exp( -C_random )
        u, v = Sinkhorn(xi_random, gQ, gR, r, r2, gamma, device=device, max_iter=max_iter, dtype=dtype)
        T = torch.diag(u) @ xi_random @ torch.diag(v)
        '''
        T,_,_ = logSinkhorn(C_random, gQ, gR, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # Use this to form the inner inverse coupling
        if r == r2:
            Lambda = torch.linalg.inv(T)
        else:
            Lambda = torch.diag(1/gQ) @ T @ torch.diag(1/gR)
    elif r == r2:
        '''
        Rank-2 initialization which requires equal inner ranks and gQ = gR = g.
        This is adapted from "Low-Rank Sinkhorn Factorization" at https://arxiv.org/pdf/2103.04737
        We advise setting full_rank = True and using the first initialization.
        '''
        g = gQ
        lambd = torch.min(torch.tensor([torch.min(a), torch.min(b), torch.min(g)])) / 2

        if rank2_random:
            # Take random sample from probability simplex
            a1 = random_simplex_sample(N1, device=device, dtype=dtype)
            b1 = random_simplex_sample(N2, device=device, dtype=dtype)
            g1 = random_simplex_sample(r, device=device, dtype=dtype)
        else:
            # or initialize exactly as in scetbon 21' ott-jax repo
            g1 = torch.arange(1, r + 1, device=device, dtype=dtype)
            g1 /= g1.sum()
            a1 = torch.arange(1, N1 + 1, device=device, dtype=dtype)
            a1 /= a1.sum()
            b1 = torch.arange(1, N2 + 1, device=device, dtype=dtype)
            b1 /= b1.sum()
        
        a2 = (a - lambd*a1)/(1 - lambd)
        b2 = (b - lambd*b1)/(1 - lambd)
        g2 = (g - lambd*g1)/(1 - lambd)
        
        # Generate Rank-2 Couplings
        Q = lambd*torch.outer(a1, g1).to(device) + (1 - lambd)*torch.outer(a2, g2).to(device)
        R = lambd*torch.outer(b1, g1).to(device) + (1 - lambd)*torch.outer(b2, g2).to(device)
        
        # This is already determined as g (but recomputed anyway)
        gR, gQ = R.T @ one_N2, Q.T @ one_N1
        
        # Last term adds very tiny off-diagonal component for the non-diagonal LC-factorization (o/w the matrix stays fully diagonal)
        T = (1-lambd)*torch.diag(g) + lambd*torch.outer(gR, gQ).to(device)
        Lambda = torch.linalg.inv(T)
    
    return Q, R, T, Lambda


def logSinkhorn(grad, a, b, gamma_k, max_iter = 50, \
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64, \
                balanced=True, unbalanced=False, \
                tau=None, tau2=None, \
                recenter_every=30, tol=1e-12, \
                squeeze=True,
               dual_1 = None, dual_2 = None):
    
    a, b = (a / a.sum()), (b / b.sum())
    
    log_a = torch.log(a)
    log_b = torch.log(b)
    
    n, m = a.size(0), b.size(0)

    if dual_1 is None and dual_2 is None:
        f_k = torch.zeros((n), device=device)
        g_k = torch.zeros((m), device=device)
    else:
        f_k = dual_1
        g_k = dual_2
    
    epsilon = gamma_k**-1
    
    if not balanced:
        ubc = (tau/(tau + epsilon ))
        if tau2 is not None:
            ubc2 = (tau2/(tau2 + epsilon ))
    
    for it in range(max_iter):
        
        f_prev = f_k.clone()
        g_prev = g_k.clone()
        
        if balanced and not unbalanced:
            # Balanced
            f_k = f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1))
            g_k = g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0))
        elif not balanced and unbalanced:
            # Unbalanced
            f_k = ubc*(f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc2*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )
        else:
            # Semi-relaxed
            f_k = (f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )
            '''
            if squeeze and it == max_iter - 1:
                f_k = (f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )'''
        
        if it % recenter_every == 0:
            # Recenter potentials; gauge invariant
            alpha = f_k.mean()
            f_k -= alpha
            g_k += alpha
        
        if max((f_k-f_prev).abs().max(), (g_k-g_prev).abs().max()) < tol:
            break
    
    P = torch.exp(Cost(f_k, g_k, grad, epsilon, device=device))

    '''
    if squeeze and (not balanced and not unbalanced):
        # Last projection of semi-relaxed always satisfies outer marginal
        P = torch.diag( a / P.sum(1) ) @ P'''
    
    return P, f_k, g_k

def Sinkhorn(xi, a, b, N1, r, gamma_k, max_iter = 50, \
             delta = 1e-9, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64):
    '''
    A lightweight impl of Sinkhorn.
    ------Parameters------
    xi: torch tensor
        An n x m matrix of the exponentiated positive Sinkhorn kernel.
    a: torch tensor
        Left outer marginal, should be positive and sum to 1.0
    b: torch tensor
        Right outer marginal, should be positive and sum to 1.0
    N1: int
        Dimension 1
    r: int
        Dimension 2
    gamma_k: float
        Step-size used for scaling convergence criterion.
    max_iter: int
        Maximum number of iterations for Sinkhorn loop
    delta: float
        Used for determining convergence to marginals
    device: str
        'cpu' if running on CPU, else 'cuda' for GPU
    dtype: torch dtype
        Defaults to float64
    '''
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        
        u_tild = u
        v_tild = v
        u = (a / (xi @ v))
        v = (b / (xi.T @ u))
        i+=1
        
    return u, v


def Delta(vark, varkm1, gamma_k):
    '''
    Convergence criterion for FRLC.
    ------Parameters------
    vark: tuple of 3-tensors
        Tuple of coordinate MD block variables (Q,R,T) at current iter
    varkm1:  tuple of 3-tensors
        Tuple of coordinate MD block variables (Q,R,T) at previous iter
    gamma_k: float
        Coordinate MD step-size
    '''
    Q, R, T = vark
    Q_prev, R_prev, T_prev = varkm1
    error = (gamma_k**-2)*(torch.norm(Q - Q_prev) + torch.norm(R - R_prev) + torch.norm(T - T_prev))
    return error


def stabilize_Q_init(Q, rand_perturb = False, 
                     lambda_factor = 0.9, max_inneriters_balanced= 300, 
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64):
    """
    Initial condition Q (e.g. from annotation, if doing a warm-start) will not optimize if one-hot.
                ---e.g. if most of Q_t is sparse/a clustering, logQ_t = - inf which is unstable!
    
    Perturb to ensure there is non-zero mass everywhere.
    """
    # Add a small random or trivial outer product perturbation to ensure stability of one-hot encoded Q
    N2, r2 = Q.shape[0], Q.shape[1]
    b, gQ = torch.sum(Q, axis = 1), torch.sum(Q, axis = 0)
    eps_Q = torch.outer(b, gQ).to(device).type(dtype)
    
    # Yield perturbation, return
    Q_init = ( 1 - lambda_factor ) * Q + lambda_factor * eps_Q
    
    return Q_init


def Cost(f, g, Grad, epsilon, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64):
    '''
    A matrix which is using for the broadcasted log-domain log-sum-exp trick-based updates.
    ------Parameters------
    f: torch.tensor (N1)
        First dual variable of semi-unbalanced Sinkhorn
    g: torch.tensor (N2)
        Second dual variable of semi-unbalanced Sinkhorn
    Grad: torch.tensor (N1 x N2)
        A collection of terms in our gradient for the update
    epsilon: float
        Entropic regularization for Sinkhorn
    device: 'str'
        Device tensors placed on
    '''
    return -( Grad - torch.outer(f, torch.ones(Grad.size(dim=1), device=device)) - torch.outer(torch.ones(Grad.size(dim=0), device=device), g) ) / epsilon
