from pathlib import Path
import ast
import dataclasses
import itertools
import json
import sys
import types
import unittest
import __future__
import numpy as np
from scipy.signal import firwin,lfilter
import experiment as ex


def original_cubic():
    path=next(Path(__file__).resolve().parent.parent.glob('01_*.ipynb'))
    nb=json.loads(path.read_text(encoding='utf-8'))
    mod=types.ModuleType('original_cubic');sys.modules[mod.__name__]=mod
    mod.__dict__.update(np=np,dataclass=dataclasses.dataclass,firwin=firwin,lfilter=lfilter,
                        combinations_with_replacement=itertools.combinations_with_replacement)
    nodes=[]
    for i in (1,3,5,7):
        for n in ast.parse(''.join(nb['cells'][i]['source'])).body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                nodes.append(n)
            elif isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id.isupper() and t.id!='OUT' for t in n.targets):
                nodes.append(n)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),mod.__dict__)
    return mod


class RevisionTests(unittest.TestCase):
    def test_generator_matches_original(self):
        old=original_cubic().generate_standalone_data()
        new=ex.generate('cubic',12345,54321,None)
        np.testing.assert_array_equal(new.clean,old.target_filtered)
        np.testing.assert_array_equal(new.lo,old.x_lo_tx)
        np.testing.assert_array_equal(new.hi,old.x_hi_tx)

    def test_dictionary_matches_direct_original(self):
        old=original_cubic();data=old.generate_standalone_data()
        x,y,n=old.build_problem(data,list(range(4)))
        new=ex.design(ex.generate('cubic',12345,54321,None),ex.terms(4,False))
        np.testing.assert_allclose(new,x[:3064],rtol=1e-10,atol=1e-17)

    def test_quintic_counts_and_permutation_symmetry(self):
        for m,total in [(2,40),(4,760),(6,4788),(8,18336)]:
            keys=ex.terms(m,True)
            self.assertEqual(len(set(keys)),total)
            for key in keys:
                if key[0]=='q5b':self.assertLessEqual(key[4],key[5])

    def test_screen_and_train_only_normalization(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(30,4))+1j*rng.normal(size=(30,4));y=x[:,0]
        unit,target,norms,ynorm=ex.normalize(x,y)
        ids,_=ex.screen(unit,target,.9)
        np.testing.assert_array_equal(ids,[0])
        self.assertEqual(len(ex.screen(unit,target,None)[0]),4)
        np.testing.assert_allclose(norms,np.linalg.norm(x,axis=0))

    def test_ridge_primal_dual_equivalence(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(8,12))+1j*rng.normal(size=(8,12));y=x[:,1]+.3j*x[:,4]
        rho=.01;h=ex.ridge(x,y,rho)
        expected=np.linalg.solve(x.conj().T@x+rho*np.sum(abs(x)**2)/12*np.eye(12),x.conj().T@y)
        np.testing.assert_allclose(h,expected,atol=1e-12)

    def test_complex_lasso_orthonormal_solution(self):
        x=np.eye(5,dtype=complex);y=np.array([1+1j,3j,0,0,0]);fraction=.1
        w,d=ex.complex_lasso(x,y,fraction)
        expected=np.maximum(0,1-.3/np.maximum(abs(y),ex.EPS))*y
        np.testing.assert_allclose(w,expected,atol=1e-11)
        self.assertTrue(d['converged'])

    def test_complex_omp(self):
        x=np.eye(5,dtype=complex);y=np.array([1+1j,3j,0,0,0])
        w,_=ex.complex_omp(x,y,2);np.testing.assert_allclose(w,y)

    def test_sbl_final_posterior_and_trace(self):
        x=np.eye(5,dtype=complex);y=np.array([.6,.8j,0,0,0])
        w,d=ex.sbl(x,y,1e6)
        self.assertTrue(d['converged'])
        active=d['active_local'];alpha=np.asarray(d['alpha'])
        expected=1e6*y[active]/(1e6+alpha)
        np.testing.assert_allclose(w[active],expected)
        self.assertTrue(all(np.isfinite(row['neg_log_evidence']) for row in d['history']))

    def test_independent_stages(self):
        a=ex.generate('tanh',20261001,20761001,35)
        b=ex.generate('tanh',20261011,20761011,35)
        self.assertFalse(np.array_equal(a.observed,b.observed))
        np.testing.assert_array_equal(a.observed,ex.generate('tanh',20261001,20761001,35).observed)

    def test_historical_cubic_control_revised_solver(self):
        record=ex.generate('cubic',12345,54321,None)
        x=ex.design(record,ex.terms(6,False),6128);y=record.clean[ex.DISCARD:]
        h,d=ex.fit(x[:3064],y[:3064],'sbl',1e6,max_iter=2000)
        self.assertTrue(d['converged'])
        self.assertEqual(d['n_active'],7)
        self.assertAlmostEqual(ex.cancellation(y[3064:],x[3064:]@h),51.2652653,places=4)

    def test_inactive_screen_ablation(self):
        r=ex.generate('tanh',12345,54321,None);keys=ex.terms(2,True)
        x=ex.design(r,keys);y=r.clean[ex.DISCARD:ex.DISCARD+3064]
        for method,parameter in [('ridge',1e-4),('sbl',1e4)]:
            a,da=ex.fit(x,y,method,parameter,None)
            b,db=ex.fit(x,y,method,parameter,.01)
            self.assertEqual(db['n_full'],db['n_screen'])
            np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
