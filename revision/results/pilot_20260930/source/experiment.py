"""Independent revision experiment. No imports from the historical archive.

Original notebooks remain frozen. This implementation caches sufficient statistics
and recomputes the final posterior after an alpha update; these deliberate solver
changes are recorded separately from historical reproduction.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import firwin, lfilter, fftconvolve

S, FS, ORDER, DISCARD = 10240, 256e6, 2048, 4112
GAIN, GAMMA, EPS = 1.2 + 3.7j, 2.0, 1e-12
PATH = np.array([.186-.023j, -.098+.117j, .291-.338j, -.607+.628j,
                 1.131-1.348j, -1.090+.993j, 1.633-1.725j, -1.415+1.365j,
                 .760-.668j, -.256+.211j, .045-.037j])


@dataclass
class Record:
    lo: np.ndarray
    hi: np.ndarray
    clean: np.ndarray
    observed: np.ndarray
    lpf: np.ndarray


def generate(source: str, tx_seed: int, noise_seed: int, pnr: float | None) -> Record:
    rng = np.random.RandomState(tx_seed)
    raw = rng.rand(S) + 1j * rng.rand(S)
    h = firwin(ORDER + 1, .05, window='hamming', scale=True)
    k, n = np.arange(ORDER + 1), np.arange(S)
    hi = lfilter(h * np.exp(1j * (2 * np.pi * 30e6 * k / FS)), [1.], raw)
    lo = lfilter(h * np.exp(1j * (2 * np.pi * -30e6 * k / FS)), [1.], raw)
    drive = lfilter(PATH, [1.], hi + lo)
    if source == 'cubic':
        nonlinear = GAIN * drive * np.abs(drive)**2
    elif source == 'tanh':
        nonlinear = GAIN * np.tanh(GAMMA*np.abs(drive)) * drive/(np.abs(drive)+EPS)
    else:
        raise ValueError(source)
    lpf = firwin(ORDER + 1, .2, window='hamming', scale=True)
    clean = lfilter(lpf, [1.], nonlinear*np.exp(1j*(2*np.pi*90e6*n/FS)))
    observed = clean.copy()
    if pnr is not None:
        variance = float(np.mean(np.abs(clean[DISCARD:])**2))*10**(-pnr/10)
        nrng = np.random.RandomState(noise_seed)
        observed += np.sqrt(variance/2)*(nrng.randn(S)+1j*nrng.randn(S))
    return Record(lo*np.exp(1j*(2*np.pi*30e6*n/FS)),
                  hi*np.exp(-1j*(2*np.pi*30e6*n/FS)), clean, observed, lpf)


def terms(memory: int, quintic: bool) -> list[tuple]:
    if memory < 1 or memory > 16:
        raise ValueError('memory must be 1..16 for the fixed transient guard')
    pairs = list(itertools.combinations_with_replacement(range(memory), 2))
    out = [('c3', a, b, c) for a, b in pairs for c in range(memory)]
    if quintic:
        out += [('q5a', a, b, c, d, e)
                for a,b,c in itertools.combinations_with_replacement(range(memory), 3)
                for d in range(memory) for e in range(memory)]
        out += [('q5b', a,b,c,d,e) for a,b in pairs
                for c in range(memory) for d,e in pairs]
    return out


def design(record: Record, keys: list[tuple], count: int = 3064) -> np.ndarray:
    if not 0 < count <= S-DISCARD:
        raise ValueError('invalid usable sample count')
    # Every requested output has the complete 2049-tap causal RX history.
    start, end = DISCARD-ORDER, DISCARD+count
    n = np.arange(start, end)
    x = np.empty((count, len(keys)), dtype=np.complex128)
    for offset in range(0, len(keys), 64):
        block = keys[offset:offset+64]
        raw = np.empty((len(n), len(block)), dtype=np.complex128)
        for j,key in enumerate(block):
            if key[0] == 'c3':
                _,a,b,c=key
                raw[:,j]=record.lo[n-a]*record.lo[n-b]*record.hi[n-c].conj()
            elif key[0] == 'q5a':
                _,a,b,c,d,e=key
                raw[:,j]=(record.lo[n-a]*record.lo[n-b]*record.lo[n-c]
                          *record.lo[n-d].conj()*record.hi[n-e].conj())
            else:
                _,a,b,c,d,e=key
                raw[:,j]=(record.lo[n-a]*record.lo[n-b]*record.hi[n-c]
                          *record.hi[n-d].conj()*record.hi[n-e].conj())
        x[:,offset:offset+len(block)] = fftconvolve(raw, record.lpf[:,None],
                                                  mode='full', axes=0)[ORDER:ORDER+count]
    return x


def cancellation(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(10*np.log10(np.mean(np.abs(y)**2)/(np.mean(np.abs(y-prediction)**2)+1e-30)))


def normalize(x: np.ndarray, y: np.ndarray):
    norms = np.linalg.norm(x, axis=0)
    ynorm = float(np.linalg.norm(y))
    if np.any(norms <= EPS) or ynorm <= EPS:
        raise ValueError('numerically zero fitting columns/target')
    return x/norms, y/ynorm, norms, ynorm


def screen(x: np.ndarray, y: np.ndarray, eta: float | None):
    scores = np.abs(x.conj().T@y)
    ids = np.arange(x.shape[1]) if eta is None else np.flatnonzero(scores > eta)
    if not len(ids):
        ids = np.array([int(np.argmax(scores))])
    return ids, scores


def ridge(x: np.ndarray, y: np.ndarray, rho: float) -> np.ndarray:
    n,p=x.shape
    penalty = rho*float(np.sum(np.abs(x)**2))/p
    if p <= n:
        gram = x.conj().T@x
        gram.flat[::p+1] += penalty
        return cho_solve(cho_factor(gram, lower=True, check_finite=False),
                         x.conj().T@y, check_finite=False)
    gram=x@x.conj().T
    gram.flat[::n+1] += penalty
    return x.conj().T@cho_solve(cho_factor(gram,lower=True,check_finite=False),y,check_finite=False)


def posterior(gram, rhs, alpha, beta):
    precision = beta*gram + np.diag(alpha)
    cf=cho_factor(precision,lower=True,check_finite=False)
    mu=beta*cho_solve(cf,rhs,check_finite=False)
    covariance=cho_solve(cf,np.eye(len(alpha)),check_finite=False)
    return mu, covariance, cf


def sbl(x, y, beta, *, max_iter=2000, alpha_init=1., alpha_max=1e6, tol=1e-6):
    gram=x.conj().T@x
    rhs=x.conj().T@y
    active=np.arange(x.shape[1]); alpha=np.full(len(active),alpha_init)
    previous=None; history=[]; converged=False
    for iteration in range(1,max_iter+1):
        mu,sigma,cf=posterior(gram,rhs,alpha,beta)
        # Complex Gaussian negative log marginal likelihood, additive N log(pi) omitted.
        residual2=float(np.vdot(y,y).real-2*np.vdot(mu,rhs).real+np.vdot(mu,gram@mu).real)
        logdet=float(2*np.log(np.diag(cf[0]).real).sum())
        objective=(-len(y)*np.log(beta)-np.log(alpha).sum()+logdet
                   +beta*np.vdot(y,y).real-beta*np.vdot(rhs,mu).real)
        updated=np.clip(np.clip(1-alpha*np.diag(sigma).real,EPS,1)/(np.abs(mu)**2+EPS),EPS,1e16)
        keep=np.isfinite(updated)&(updated<=alpha_max)
        if not keep.any():keep[np.argmin(updated)]=True
        rel=None if previous is None else float(np.linalg.norm(mu-previous)/max(np.linalg.norm(previous),EPS))
        alpha_rel=float(np.linalg.norm(updated-alpha)/max(np.linalg.norm(alpha),EPS))
        history.append({'iteration':iteration,'active':len(active),'fit_residual_power':max(residual2,0)/len(y),
                        'neg_log_evidence':float(objective),'mu_relative_change':rel,
                        'alpha_relative_change':alpha_rel,'pruned':int((~keep).sum())})
        if not keep.all():
            gram=gram[np.ix_(keep,keep)];rhs=rhs[keep];active=active[keep]
            alpha=updated[keep];previous=None
            continue
        alpha=updated
        if rel is not None and rel<tol:
            converged=True;break
        previous=mu.copy()
    # Explicit revised variant: final mean and alpha belong to the same posterior.
    mu,_,_=posterior(gram,rhs,alpha,beta)
    weights=np.zeros(x.shape[1],complex);weights[active]=mu
    return weights, {'converged':converged,'iterations':iteration,'history':history,
                     'last_mu_relative_change':history[-1]['mu_relative_change'],
                     'alpha':alpha.tolist(),'active_local':active.tolist()}


def complex_lasso(x,y,fraction,*,max_iter=10000,tol=1e-6):
    """Minimize .5||y-Xw||^2 + lambda sum(abs(complex w)); no intercept."""
    gram=x.conj().T@x; rhs=x.conj().T@y
    # Absolute row-sum bounds the largest eigenvalue; safe deterministic step.
    lipschitz=float(np.max(np.sum(np.abs(gram),axis=1)))
    lam=fraction*float(np.max(np.abs(rhs)))
    w=np.zeros(x.shape[1],complex);z=w.copy();t=1.;converged=False;kkt=float('inf')
    for iteration in range(1,max_iter+1):
        v=z-(gram@z-rhs)/lipschitz
        new=np.maximum(0,1-lam/(lipschitz*np.maximum(np.abs(v),EPS)))*v
        tn=(1+np.sqrt(1+4*t*t))/2
        z=new+(t-1)/tn*(new-w); w=new;t=tn
        if iteration%20==0 or iteration==max_iter:
            gradient=gram@w-rhs;nonzero=np.abs(w)>EPS
            violation=np.maximum(np.abs(gradient)-lam,0)
            violation[nonzero]=np.abs(gradient[nonzero]+lam*w[nonzero]/np.abs(w[nonzero]))
            kkt=float(violation.max()/max(np.abs(rhs).max(),EPS))
            if kkt<tol:converged=True;break
    return w,{'converged':converged,'iterations':iteration,'kkt_relative':kkt}


def complex_omp(x,y,count):
    active=[];residual=y.copy(); w=np.zeros(x.shape[1],complex)
    for _ in range(min(count,x.shape[1],len(y))):
        score=np.abs(x.conj().T@residual)
        score[active]=-1
        idx=int(score.argmax())
        if score[idx]<=EPS:break
        active.append(idx)
        coef=np.linalg.lstsq(x[:,active],y,rcond=None)[0]
        residual=y-x[:,active]@coef
    if active:w[active]=coef
    return w,{'converged':True,'iterations':len(active)}


def fit(x,y,method,parameter,eta=None,*,max_iter=2000,alpha_init=1.):
    begin=time.perf_counter()
    unit,target,norms,ynorm=normalize(x,y)
    selected,scores=screen(unit,target,eta)
    t_screen=time.perf_counter()-begin
    if method=='ridge':
        # Preserve the historical scale-adaptive penalty in physical coordinates.
        weights=ridge(x[:,selected],y,parameter)
        details={'converged':True,'iterations':1}
    else:
        a=unit[:,selected]
        if method=='sbl':w,details=sbl(a,target,parameter,max_iter=max_iter,alpha_init=alpha_init)
        elif method=='lasso':w,details=complex_lasso(a,target,parameter)
        elif method=='omp':w,details=complex_omp(a,target,int(parameter))
        else:raise ValueError(method)
        weights=ynorm*w/norms[selected]
    h=np.zeros(x.shape[1],complex);h[selected]=weights
    details.update({'fit_sec':time.perf_counter()-begin,'screen_sec':t_screen,
                    'n_full':x.shape[1],'n_screen':len(selected),'n_active':int(np.count_nonzero(h)),
                    'score_min':float(scores.min()),'score_max':float(scores.max())})
    return h,details
