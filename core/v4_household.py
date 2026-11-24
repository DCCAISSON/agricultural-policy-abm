"""V4 estimation and batch planning on the existing annual area economy."""
from hashlib import sha256

import numpy as np
from scipy.special import expit, softmax

FIELD_NAMES = ('share', 'log_area', 'workers', 'log_liquidity', 'transfer')
DIRECTIONS = ('decrease_grain', 'maintain', 'increase_grain')


def uniforms(seed, year, ids):
    return np.array([[int.from_bytes(sha256(f'v4:{stream}:{seed}:{year}:{i}'.encode()).digest()[:8],
                                    'big') / 2**64 for stream in ('stay', 'direction', 'amount', 'out', 'in')]
                     for i in ids])


def features(share, area, workers, liquidity, transfer, region, regions, year):
    # Earlier years' dummies are controls, not a forecast of future year effects.
    x = np.column_stack((np.ones(len(area)), share, np.log1p(area), workers / 3.,
                         np.log1p(liquidity) / 10., transfer,
                         *(region == r for r in regions[1:]),
                         np.broadcast_to(year, len(area)) == 2011,
                         np.broadcast_to(year, len(area)) == 2012))
    return x.astype(float)


def repair_features(prior, area, workers, transfer, region, regions, year):
    """Use the same non-cash covariates for estimation and simulation."""
    represented=np.isfinite(prior)
    x=features(np.where(represented,prior,0.),area,workers,np.zeros(len(area)),transfer,region,regions,year)
    return np.column_stack((np.delete(x,4,axis=1),represented))


def fit_categories(x, y, cluster, classes):
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from scipy.linalg import pinvh
    _, inverse = np.unique(cluster, return_inverse=True)
    p = x.shape[1]
    target = np.eye(classes)[y]
    ridge = 1.

    def objective(beta):
        utilities = np.column_stack((np.zeros(len(y)), x @ beta.reshape(classes-1, p).T))
        log_p = utilities - logsumexp(utilities, axis=1, keepdims=True)
        prob = np.exp(log_p)
        loss = -(target * log_p).sum() + .5 * ridge * (beta @ beta)
        gradient = ((prob-target)[:, 1:].T @ x).ravel() + ridge * beta
        return loss, gradient

    fit = minimize(objective, np.zeros((classes-1)*p), jac=True, method='L-BFGS-B',
                   options={'maxiter': 1800, 'ftol': 1e-9})
    if not fit.success:
        raise RuntimeError(f'NFPS classifier: {fit.message}')
    coef = np.vstack((np.zeros(p), fit.x.reshape(classes-1, p)))
    pr = softmax(x @ coef.T, axis=1)[:, 1:]
    scores = ((target[:, 1:]-pr)[:, :, None] * x[:, None, :]).reshape(len(x), -1)
    grouped = np.zeros((inverse.max()+1, scores.shape[1]))
    np.add.at(grouped, inverse, scores)
    hess = np.block([[(x.T*(pr[:, i]*((i == j)-pr[:, j]))) @ x
                       for j in range(classes-1)] for i in range(classes-1)])
    inv = pinvh(hess + ridge * np.eye(hess.shape[0]))
    sandwich = (grouped @ inv).T @ (grouped @ inv)
    return coef, np.vstack((np.zeros(p), np.sqrt(sandwich.diagonal()).reshape(classes-1, p)))


def farm_batch(state, fit, u, *, stage, trade=False, incentive=1., grain_aid=0., diagnostic=None):
    """Three feasible physical plans, followed by one probability draw per household."""
    land, rights, funds, rent, intensity, labor, workers, transfer, prior, region = (
        np.asarray(state[k]) for k in ('land','rights','funds','rent','intensity','labor',
                                       'workers','transfer','prior','region'))
    revenue_g, revenue_c, cost_g, cost_c, labor_g, labor_c, wage, aid = np.asarray(state['inputs']).T
    bounds = np.array([5.,15.,30.])
    repaired='entry_shares' in fit
    entry=~np.isfinite(prior)
    if repaired:
        x=repair_features(prior,land,workers,transfer,region,fit['regions'],0)
        init_group=np.digitize(land,bounds)
        guess=np.zeros(len(prior))
        for i in np.flatnonzero(entry):
            donor=fit['entry_shares'].get((region[i],int(init_group[i])),fit['entry_shares']['all'])
            guess[i]=donor[min(int(u[i,2]*len(donor)),len(donor)-1)]
        prior=np.where(entry,guess,prior)
    else:
        prior = np.where(np.isfinite(prior), prior, fit['prior_median'])
        x = features(prior, land, workers, funds, transfer, region, fit['regions'], 0)
    group = 3*np.digitize(land,bounds)+np.digitize(prior,[1e-8,1.-1e-8])
    ep=fit['endpoint'][group]
    draw=np.clip((u[:,2,None]-ep)/np.maximum(1e-10,1.-ep),0.,1.)
    magnitudes = fit['amplitudes'][group[:,None],np.arange(2),np.minimum((draw*19).astype(int),18)]
    magnitudes=np.where(u[:,2,None]<ep,1.,magnitudes)
    shares = np.column_stack((prior*(1-magnitudes[:,0]),prior,prior+(1.-prior)*magnitudes[:,1]))
    if repaired:
        # New and re-entering farms have a drawn initial composition, not a synthetic incumbent.
        shares[entry]=prior[entry,None]
    material = cost_c[:,None] + (cost_g-cost_c)[:,None]*shares
    days = labor_c[:,None] + (labor_g-labor_c)[:,None]*shares
    sales = revenue_c[:,None] + (revenue_g+grain_aid-revenue_c)[:,None]*shares
    # Cost(S) = material*S + wage*max(0, days*S-family time). Both linear inequalities bind.
    full = np.maximum(0.,land)*intensity
    cash_cap = np.minimum(funds[:,None]/material,
        (funds+wage*labor)[:,None]/(material+wage[:,None]*days))
    family_cusp = labor[:,None]/days
    flags=diagnostic or {}
    if trade:
        can_buy = (u[:,4] < expit(x @ fit['rent_in'][1])) & (rent > 0.) if flags.get('rent_in',stage>=2) else rent>0.
        marginal_after_hires = sales-material-wage[:,None]*days-rent[:,None]/intensity[:,None]
        # The rent-bearing tranche is feasible only after funding retained land first.
        unbounded = np.maximum(land,np.asarray(state['limit']))*intensity
        rent_sown = rent/intensity
        with_rent = np.minimum((funds+rent_sown*full)[:,None]/(material+rent_sown[:,None]),
            (funds+rent_sown*full+wage*labor)[:,None]/(material+rent_sown[:,None]+wage[:,None]*days))
        grow = np.where(marginal_after_hires>0.,with_rent,np.minimum(with_rent,family_cusp))
        cash_cap = np.where(can_buy[:,None] & (grow>full[:,None]),np.minimum(unbounded[:,None],grow),cash_cap)
        total_limit=np.where(can_buy[:,None]& (grow>full[:,None]),unbounded[:,None],full[:,None])
    else:
        total_limit=full[:,None]
    q = np.minimum(total_limit,cash_cap)
    # On either side of the family-labor breakpoint, the objective is linear.
    sizes = np.stack((np.zeros_like(q),np.minimum(q,family_cusp),np.minimum(q,full[:,None]),q),axis=2)
    planted = sizes/intensity[:,None,None]
    hired = np.maximum(0.,days[:,:,None]*sizes-labor[:,None,None])
    cost = material[:,:,None]*sizes+wage[:,None,None]*hired
    lease_cost = rent[:,None,None]*np.maximum(0.,planted-land[:,None,None]) if trade else 0.
    value = (sales[:,:,None]*sizes-cost-lease_cost+
             aid[:,None,None]*np.minimum(rights[:,None,None],planted))
    # Evaluate the eligibility kink separately; P0 is paid after harvest.
    p = np.minimum(q,rights[:,None]*intensity[:,None])
    h = np.maximum(0.,days*p-labor[:,None])
    pc=material*p+wage[:,None]*h
    pv=sales*p-pc+aid[:,None]*p/intensity[:,None]
    option=np.argmax(value,axis=2)[:,:,None]
    def selected(array):
        return np.take_along_axis(array,option,axis=2)[:,:,0]
    best = selected(value)
    use_kink = pv>best
    total = np.where(use_kink,p,selected(sizes))
    hired = np.where(use_kink,h,selected(hired))
    cost = np.where(use_kink,pc,selected(cost))
    best = np.maximum(best,pv)
    slack = np.maximum(0.,labor[:,None]-days*total)/180.
    liquid = np.log1p(np.maximum(0.,funds[:,None]-cost))/10.
    if repaired:
        ctype=np.digitize(prior,[1e-8,1.-1e-8])
        scores=np.einsum('ij,ikj->ik',x,fit['direction_by_prior'][ctype])
    else:
        scores = x @ fit['direction'].T
    scores += incentive*(best-best[:,1,None])/np.maximum(1.,full[:,None])/fit['return_scale']
    if flags.get('scores',stage>=2):
        scores += .25 * ((slack-slack[:,1,None]) + (liquid-liquid[:,1,None]))
    feasible = (np.abs(shares-prior[:,None])>1e-8) & (total>1e-8)
    feasible[:,1] = True
    probabilities = softmax(np.where(feasible,scores,-np.inf),axis=1)
    idx = (u[:,1,None]>np.cumsum(probabilities,axis=1)[:,:-1]).sum(axis=1)
    i = np.arange(len(idx))
    if flags.get('stay',stage>=2):
        if repaired:
            beta=np.where(np.asarray(state['had_operating_land'])[:,None],fit['stay_operator'][1],fit['enter_landless'][1])
            p=np.sum(x*beta,axis=1)
        else:
            p=x@fit['stay'][1]
        participation = expit(p+incentive*np.max(best,axis=1)/np.maximum(1.,full)/fit['return_scale'])
    else:
        participation = np.ones(len(idx))
    # V4.1 keeps economic exit (zero is already a candidate). For a landless
    # household, pre-trade area is only a bid; post-trade area requires an actual lease.
    chosen = total[i,idx] if flags.get('economic_participation') else total[i,idx]*(u[:,0]<participation)
    g, c = chosen*shares[i,idx], chosen*(1.-shares[i,idx])
    phys=chosen/intensity
    ask=np.maximum(0.,(sales[i,idx]-material[i,idx]-
        wage*days[i,idx]*(chosen*days[i,idx]>=labor-1e-8))*intensity)
    can_sell = (u[:,3]<expit(x@fit['rent_out'][1])) | (chosen<=1e-8) if flags.get('rent_out',stage>=2) else np.ones(len(idx),dtype=bool)
    offer=np.where(can_sell,
                   np.minimum(rights,np.maximum(0.,land-phys)),0.) if trade else np.zeros(len(idx))
    demand=np.maximum(0.,phys-land) if trade else np.zeros(len(idx))
    bid=np.divide(np.maximum(0.,funds-cost[i,idx]),demand,out=np.zeros_like(phys),where=demand>1e-8)
    # The plan values retained land at its marginal crop return, exclusive of a nonportable subsidy.
    bid=np.minimum(bid,np.maximum(0.,(sales[i,idx]-material[i,idx]-
        wage*days[i,idx]*(chosen*days[i,idx]>labor+1e-8))*intensity))
    record=dict(grain=g,crop=c,cultivated=phys,eligible=np.minimum(rights,phys),
                hired=np.maximum(0.,chosen*days[i,idx]-labor),
                cost=chosen*material[i,idx]+wage*np.maximum(0.,chosen*days[i,idx]-labor),
                family=np.minimum(labor,chosen*days[i,idx]),offer=offer,demand=demand,ask=ask,bid=bid,
                stay_probability=participation,direction=idx,probabilities=probabilities,
                shadow_cash=chosen>=cash_cap[i,idx]-1e-7, value=best[i,idx],
                prior=prior, target=shares[i,idx])
    if flags.get('trace'):
        record.update(stay_u=u[:,0],plan_sown=total[i,idx],plan_land=total[i,idx]/intensity,
            max_sown_land=total_limit[np.arange(len(idx)),idx] if total_limit.shape[1]==3 else total_limit[:,0],
            max_sown_affordable=q[i,idx],plan_cash=cost[i,idx],plan_hired=hired[i,idx],
            plan_labor=total[i,idx]*days[i,idx],classifier_funds=funds,
            labor_binding=total[i,idx]*days[i,idx]>=labor-1e-8,
            land_binding=(total[i,idx]>=full-1e-7)&(total[i,idx]>1e-7))
    return record
