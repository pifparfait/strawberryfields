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

class TestBosonicBackend:
    r"""Tests some methods of the BosonicBackend class which
    are not covered by the standard backends tests."""
    
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
            assert state.num_weights == 4
            assert state.weights().shape == (4,)
            assert np.allclose(sum(state.weights()),1)
            assert state.means().shape == (4,2)
            assert state.covs().shape == (4,2,2)
            covs_compare = np.tile(np.eye(2),(4,1,1))
            assert np.allclose(state.covs(), covs_compare)
        else:
            assert state.num_weights == 1
            assert state.weights().shape == (1,)
            assert np.allclose(sum(state.weights()),1)
            assert state.means().shape == (1,2)
            assert state.covs().shape == (1,2,2)
            covs_compare = np.tile(np.eye(2),(1,1,1))
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
    def test_cat_state_representations(self,alpha,phi):
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
        
        
        
        
        
        