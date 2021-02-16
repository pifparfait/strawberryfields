# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Unit tests for backends.bosonicbackend.bosoniccircuit.py.
"""

import numpy as np
import strawberryfields as sf
import strawberryfields.backends.bosonicbackend.backend as bosonic
import pytest

pytestmark = pytest.mark.bosonic

ALPHA_VALS = np.linspace(-1,1,5)
PHI_VALS = np.linspace(0,1,3)
FOCK_VALS = np.arange(5,dtype=int)
r_fock = 0.05  
EPS_VALS = np.array([0.01,0.05,0.1,0.5])



class TestBosonicCatStates:
    r"""Tests cat state method of the BosonicBackend class."""
    
    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_cat_complex(self,alpha,phi):
        r"""Checks the complex cat state representation."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha,phi) | q[0]
              
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        state = backend.state()
        
       
        if alpha != 0:
            #Check shapes
            assert state.num_weights == 4
            assert state.weights().shape == (4,)
            assert np.allclose(sum(state.weights()),1)
            assert state.means().shape == (4,2)
            assert state.covs().shape == (4,2,2)
            covs_compare = np.tile(np.eye(2)*sf.hbar/2,(4,1,1))
            assert np.allclose(state.covs(), covs_compare)
            
            #Weights not real if phi != 0 or 1
            if phi % 1 == 0:
                assert np.allclose(state.weights().real, state.weights())
            else:
                assert not np.allclose(state.weights().real, state.weights())
            # Covs should be real, means complex
            assert not np.allclose(state.means().real, state.means())
            assert np.allclose(state.covs().real, state.covs())
        else:
            assert state.num_weights == 1
            assert state.weights().shape == (1,)
            assert np.allclose(sum(state.weights()),1)
            assert state.means().shape == (1,2)
            assert state.covs().shape == (1,2,2)
            covs_compare = np.tile(np.eye(2)*sf.hbar/2,(1,1,1))
            assert np.allclose(state.covs(), covs_compare)
    
    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_cat_real(self,alpha,phi):
        r"""Checks the real cat state representation."""
        # Check that low cutoff and low D produce fewer weights when alpha !=0
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha,phi,cutoff=1e-6,desc="real",D=1) | q[0]
              
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        state = backend.state()
        num_weights_low = state.num_weights
        
        assert np.allclose(sum(state.weights()),1)
        assert state.means().shape == (num_weights_low,2)
        assert state.covs().shape == (num_weights_low,2,2)
        
        # Weights, means and covs should be real
        assert np.allclose(state.weights().real, state.weights())
        assert np.allclose(state.means().real, state.means())
        assert np.allclose(state.covs().real, state.covs())
        
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Catstate(alpha,phi,cutoff=1e-12,desc="real",D=10) | q[0]
              
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        state = backend.state()
        num_weights_high = state.num_weights
        
        assert np.allclose(sum(state.weights()),1)
        assert state.means().shape == (num_weights_high,2)
        assert state.covs().shape == (num_weights_high,2,2)
        
        if alpha != 0:
            assert num_weights_low < num_weights_high
        else:
            assert num_weights_low == num_weights_high
    
    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    @pytest.mark.parametrize("phi", PHI_VALS)
    def test_cat_state_wigners(self,alpha,phi):
        r"""Checks that the real and complex cat state representations
        have the same Wigner functions as the cat state from the Fock 
        backend."""
        x = np.linspace(-2*alpha,2*alpha,100)
        p = np.linspace(-2*alpha,2*alpha,100)
        
        prog_complex_cat = sf.Program(1)
        with prog_complex_cat.context as qc:
            sf.ops.Catstate(alpha,phi) | qc[0]
        
        prog_real_cat = sf.Program(1)
        with prog_real_cat.context as qr:
            sf.ops.Catstate(alpha,phi,desc="real",D=10) | qr[0]
        
        backend_complex = bosonic.BosonicBackend()
        backend_complex.run_prog(prog_complex_cat,1)
        wigner_complex = backend_complex.state().wigner(0,x,p)
        
        backend_real = bosonic.BosonicBackend()
        backend_real.run_prog(prog_real_cat,1)
        wigner_real = backend_real.state().wigner(0,x,p)
        
        prog_cat_fock = sf.Program(1)
        with prog_cat_fock.context as qf:
            if alpha != 0:
                sf.ops.Catstate(alpha,phi) | qf[0]
            else:
                sf.ops.Vacuum() | qf[0]
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 20})
        results = eng.run(prog_cat_fock)
        wigner_fock = results.state.wigner(0,x,p)
        
        assert np.allclose(wigner_complex,wigner_real,rtol=1e-3,atol=1e-6)
        assert np.allclose(wigner_complex,wigner_fock,rtol=1e-3,atol=1e-6)
        assert np.allclose(wigner_fock,wigner_real,rtol=1e-3,atol=1e-6)

    @pytest.mark.parametrize("alpha", ALPHA_VALS)
    def test_cat_state_parity(self,alpha):
        r"""Checks that the real and complex cat state representations
        yield the correct parity."""    
        # for phi = 0, should yield parity of 1
        prog_complex_cat = sf.Program(1)
        with prog_complex_cat.context as qc:
            sf.ops.Catstate(alpha) | qc[0]
        
        prog_real_cat = sf.Program(1)
        with prog_real_cat.context as qr:
            sf.ops.Catstate(alpha,desc="real") | qr[0]
        
        backend_complex = bosonic.BosonicBackend()
        backend_complex.run_prog(prog_complex_cat,1)
        state_complex = backend_complex.state()
        parity_complex = state_complex.parity_expectation([0])
        
        backend_real = bosonic.BosonicBackend()
        backend_real.run_prog(prog_real_cat,1)
        state_real = backend_real.state()
        parity_real = state_real.parity_expectation([0])
               
        assert np.allclose(parity_complex, 1)
        assert np.allclose(parity_real, 1)
        
        # for phi = 1, should yield parity of -1 unless alpha == 0
        if alpha != 0:
            prog_complex_cat = sf.Program(1)
            with prog_complex_cat.context as qc:
                sf.ops.Catstate(alpha,1) | qc[0]
            
            prog_real_cat = sf.Program(1)
            with prog_real_cat.context as qr:
                sf.ops.Catstate(alpha,1,desc="real") | qr[0]
            
            backend_complex = bosonic.BosonicBackend()
            backend_complex.run_prog(prog_complex_cat,1)
            state_complex = backend_complex.state()
            parity_complex = state_complex.parity_expectation([0])
            
            backend_real = bosonic.BosonicBackend()
            backend_real.run_prog(prog_real_cat,1)
            state_real = backend_real.state()
            parity_real = state_real.parity_expectation([0])
                   
            assert np.allclose(parity_complex, -1)
            assert np.allclose(parity_real, -1)

     
class TestBosonicFockStates:
    r"""Tests fock state method of the BosonicBackend class."""
    
    @pytest.mark.parametrize("n", FOCK_VALS)
    def test_fock(self,n):
        r"""Checks fock states in the bosonic representation."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Fock(n) | q[0]
              
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        state = backend.state()
        
        #Check shapes
        assert state.num_weights == n+1
        assert state.weights().shape == (n+1,)
        assert np.allclose(sum(state.weights()),1)
        assert state.means().shape == (n+1,2)
        assert state.covs().shape == (n+1,2,2)
        
        #Check mean photon is close to n
        mean,var = state.mean_photon(0)
        assert np.allclose(mean,n,atol=r_fock)
        assert np.allclose(var,0,atol=r_fock)
        
        # Weights, means and covs should be real
        assert np.allclose(state.weights().real, state.weights())
        assert np.allclose(state.means().real, state.means())
        assert np.allclose(state.covs().real, state.covs())
        
        if n == 0:
            covs_compare = np.tile(np.eye(2),(1,1,1))
            assert np.allclose(state.covs(), covs_compare)
            assert np.allclose(state.fidelity_vacuum(),1)
    
    @pytest.mark.parametrize("n", FOCK_VALS)
    def test_fock_state_wigners(self,n):
        r"""Checks that fock state Wigner functions in the bosonic and Fock
        backends match."""
        x = np.linspace(-n,n,100)
        p = np.linspace(-n,n,100)
        
        prog = sf.Program(1)
        with prog as q:
            sf.ops.Fock(n) | q[0]
        
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        wigner_bosonic = backend.state().wigner(0,x,p)
        
        eng = sf.Engine("fock", backend_options={"cutoff_dim": int(n+1)})
        results = eng.run(prog)
        wigner_fock = results.state.wigner(0,x,p)
        
        assert np.allclose(wigner_fock,wigner_bosonic,atol=r_fock)

    @pytest.mark.parametrize("n", FOCK_VALS)
    def test_fock_state_parity(self,n):
        r"""Checks that fock states have the right parity."""    
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.Fock(n) | q[0]
              
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        state = backend.state()
        assert np.allclose(state.parity_expectation([0]),(-1.0)**n,atol=r_fock)

class TestBosonicGKPStates:
    r"""Tests gkp method of the BosonicBackend class."""
    
    @pytest.mark.parametrize("eps", EPS_VALS)
    def test_gkp(self,eps):
        r"""Checks the complex cat state representation."""
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.GKP(epsilon=eps) | q[0]
              
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        state = backend.state()
        num_weights = state.num_weights
        assert state.weights().shape == (num_weights,)
        assert np.allclose(sum(state.weights()),1)
        assert state.means().shape == (num_weights,2)
        assert state.covs().shape == (num_weights,2,2)
        
        # Weights, means and covs should be real
        assert np.allclose(state.weights().real, state.weights())
        assert np.allclose(state.means().real, state.means())
        assert np.allclose(state.covs().real, state.covs())
        
        # Covariance should be diagonal with these entries
        cov_val = (1 - np.exp(-2 * eps))/ (1 + np.exp(-2 * eps))*sf.hbar/2
        covs_compare = np.tile(cov_val*np.eye(2),(num_weights,1,1))
        assert np.allclose(state.covs(),covs_compare)
        
        # Means should be integer multiples of sqrt(pi*hbar)/2 times a damping
        damping = 2 * np.exp(-eps) / (1 + np.exp(-2 * eps))
        mean_ints = state.means()/(damping*np.sqrt(np.pi*sf.hbar)/2)
        # Round to make sure numerical errors are washed out
        mean_ints = np.round(np.real_if_close(mean_ints),10)
        assert np.allclose(mean_ints % 1, np.zeros(mean_ints.shape))
    
    @pytest.mark.parametrize("eps", EPS_VALS)
    def test_gkp_logical(self,eps):
        r"""Checks that logically equivalent GKP states have
        Wigner functions that agree."""
        x = np.linspace(-3*np.sqrt(np.pi),3*np.sqrt(np.pi),40)
        p = np.linspace(-3*np.sqrt(np.pi),3*np.sqrt(np.pi),40)
        
        # Prepare GKP 0 and apply Hadamard
        prog_0H = sf.Program(1)
        with prog_0H.context as q0:
            sf.ops.GKP(epsilon=eps) | q0[0]
            sf.ops.Rgate(np.pi/2) | q0[0]
              
        backend_0H = bosonic.BosonicBackend()
        backend_0H.run_prog(prog_0H,1)
        state_0H = backend_0H.state()
        wigner_0H = state_0H.wigner(0,x,p)
        
        # Prepare GKP +
        prog_plus = sf.Program(1)
        with prog_plus.context as qp:
            sf.ops.GKP(state=[np.pi/2,0],epsilon=eps) | qp[0]
              
        backend_plus = bosonic.BosonicBackend()
        backend_plus.run_prog(prog_plus,1)
        state_plus = backend_plus.state()
        wigner_plus = state_plus.wigner(0,x,p)
        
        assert np.allclose(wigner_0H,wigner_plus)
    
    @pytest.mark.parametrize("eps", EPS_VALS)
    def test_gkp_state_parity(self,eps):
        r"""Checks that GKP states yield the right parity."""    
        prog = sf.Program(1)
        with prog.context as q:
            sf.ops.GKP(epsilon=eps) | q[0]
              
        backend = bosonic.BosonicBackend()
        backend.run_prog(prog,1)
        state = backend.state()
        assert np.allclose(state.parity_expectation([0]),1,atol=r_fock)
        
        
        
        