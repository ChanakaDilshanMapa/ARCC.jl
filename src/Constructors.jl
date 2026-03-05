export make_slices, make_projectors

function make_slices(C, T, Tbar, nocc, nbasis)::Slices
   Slices(
      C_occ(C, nocc),
      C_vir(C, nocc, nbasis),
      T_occ(T, nocc),
      T_vir(T, nocc, nbasis),
      Tbar_occ(Tbar, nocc),
      Tbar_vir(Tbar, nocc, nbasis)
   )   
end

function make_projectors(slice::Slices)::Projectors
   Projectors(
      ao_projector(slice::Slices),
      complementary_projector(slice::Slices),
      ao_projector_bar(slice::Slices),
      complementary_projector_bar(slice::Slices)
   )   
end

export make_physaoeris, make_integrals, make_coulomb_integrals

function make_physaoeris(aoeris)::PhysAOEris
   PhysAOEris(
      phys_aoeris(aoeris)
   )
end

function make_integrals(proj::Projectors, peris::PhysAOEris)::Integrals
   Integrals(
      piintegrals(proj::Projectors, peris::PhysAOEris),
      a_integral(proj::Projectors, peris::PhysAOEris),
      b_integral(proj::Projectors, peris::PhysAOEris),
      c_integral(proj::Projectors, peris::PhysAOEris),
      c_integral_permuted(proj::Projectors, peris::PhysAOEris),
      d_integral(proj::Projectors, peris::PhysAOEris)
   )
end

function make_coulomb_integrals(int::Integrals, slice::Slices)::CoulombIntegrals
   CoulombIntegrals(
      j_integral(int::Integrals, slice::Slices),
      k_integral(int::Integrals, slice::Slices),
      j_integralθ(int::Integrals),
      k_integralθ(int::Integrals)
   )
end

export make_fock_operators, make_fock_diags_and_offs, make_fock_integrals

function make_fock_operators(proj::Projectors, fock_ao)::FockOperators
   FockOperators(
      fock_o(proj::Projectors, fock_ao),
      fock_v(proj::Projectors, fock_ao)
   )
end

function make_fock_diags_and_offs(fop::FockOperators)::FockDiagOffs
   FockDiagOffs(
      fock_o_diag(fop::FockOperators),
      fock_v_diag(fop::FockOperators),
      fock_o_off(fop::FockOperators),
      fock_v_off(fop::FockOperators)
   )
end

function make_fock_integrals(fop::FockOperators, int::Integrals)::FockIntegrals
   FockIntegrals(
      g_o(fop::FockOperators, int::Integrals),
      g_v(fop::FockOperators, int::Integrals)
   )
end

export make_fixed_point_elements

function make_fixed_point_elements(int::Integrals, coulombint::CoulombIntegrals, slice::Slices, fd::FockDiagOffs, nbasis)::FixedPointElements
   FixedPointElements(
      bigR(int::Integrals, coulombint::CoulombIntegrals, slice::Slices),
      deltaF(fd::FockDiagOffs, slice::Slices),
      ao_denominator(fd::FockDiagOffs, nbasis)
   )   
end