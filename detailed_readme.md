# ADMDynAnlz Scientific README

ADMDynAnlz is a workflow generator for molecular-dynamics post-processing. The GUI collects file patterns, physical parameters, precision choices, and execution settings, then writes Python/VMD workflows for trajectory-derived observables. The practical controls are described in the [manual](Manual.md); this document summarizes the scientific definitions used by the software and points each concept back to its manual section.

## Shared Conventions

Most modules use a `Base Directory`, common-term pattern expansion, optional trajectory selection, and explicit precision settings. These workflow conventions are described in [General Concepts](Manual.md#general-concepts), [Base Directory](Manual.md#base-directory), [File Patterns](Manual.md#file-patterns), [Common Terms](Manual.md#common-terms), and [Precision](Manual.md#precision).

Coordinates are treated as Cartesian vectors. When periodic boundary conditions are used, minimum-image displacement components follow the usual orthorhombic convention:

$$
{\Huge
\Delta \mathbf r =
\mathbf r_i-\mathbf r_j
-\mathbf L\,\mathrm{round}\!\left(
\frac{\mathbf r_i-\mathbf r_j}{\mathbf L}
\right)
}
$$

where \(\mathbf L=(L_x,L_y,L_z)\). Static-structure-factor real-space cutoffs and RDF pair distances use the same minimum-image geometry when box dimensions are provided; see [Module 6: Static Structure Factor](Manual.md#module-6-static-structure-factor) and [Extra Module: Radial Distribution Function](Manual.md#extra-module-radial-distribution-function).

## Coordinates and Centers of Mass

The coordinate extraction module prepares raw trajectory data, continuous coordinates, and center-of-mass trajectories; see [Module 1: Coordinates Extraction](Manual.md#module-1-coordinates-extraction), [Step 1: Extraction](Manual.md#step-1-extraction), [Step 2: Continuous Coordinates](Manual.md#step-2-continuous-coordinates), and [Step 3: COM Calculation](Manual.md#step-3-com-calculation).

For a selected group of atoms, the center of mass is:

$$
{\Huge
\mathbf R_{\mathrm{COM}} =
\frac{\sum_i m_i\mathbf r_i}{\sum_i m_i}
}
$$

When continuous coordinates are generated, the intended scientific object is an unwrapped trajectory in which frame-to-frame displacements are continuous rather than folded at the periodic boundary. The conceptual update is:

$$
{\Huge
\mathbf R(t_n) =
\mathbf R(t_{n-1}) +
\left[
\mathbf r(t_n)-\mathbf r(t_{n-1})
-\mathbf L\,\mathrm{round}\!\left(
\frac{\mathbf r(t_n)-\mathbf r(t_{n-1})}{\mathbf L}
\right)
\right]
}
$$

The `Fix 1st Frame` option repairs molecules already split in the first frame before the normal unwrapping pass; see [Step 2: Continuous Coordinates](Manual.md#step-2-continuous-coordinates).

## Velocities and Dipoles

Velocity extraction and dipole calculations are described in [Module 2: Velocities and Dipoles](Manual.md#module-2-velocities-and-dipoles), [Velocity Extraction Tab](Manual.md#velocity-extraction-tab), and [Dipole Calculations Tab](Manual.md#dipole-calculations-tab).

The center-of-mass velocity for a group is:

$$
{\Huge
\mathbf V_{\mathrm{COM}} =
\frac{\sum_i m_i\mathbf v_i}{\sum_i m_i}
}
$$

For individual dipoles, the molecular dipole moment is computed from atomic charges relative to the selected reference point, usually the molecular COM:

$$
{\Huge
\boldsymbol\mu =
\sum_i q_i\left(\mathbf r_i-\mathbf R_{\mathrm{ref}}\right)
}
$$

With this sign convention, the dipole moment vector points from the negative pole toward the positive pole. The software can write either vector components or magnitudes:

$$
{\Huge
|\boldsymbol\mu| =
\sqrt{\mu_x^2+\mu_y^2+\mu_z^2}
}
$$

## MSD and Non-Gaussian Parameters

The MSD and NGP workflows are described in [Module 3: MSD and NGP / Anisotropic NGP](Manual.md#module-3-msd-and-ngp--anisotropic-ngp) and [alpha calculation settings](Manual.md#α₂t--α_anit-calculation).

For a lag time \(t\), the displacement is:

$$
{\Huge
\Delta \mathbf r(t;t_0) =
\mathbf r(t_0+t)-\mathbf r(t_0)
}
$$

The mean-squared displacement is:

$$
{\Huge
\mathrm{MSD}(t) =
\left\langle |\Delta \mathbf r(t)|^2 \right\rangle
}
$$

The standard three-dimensional non-Gaussian parameter is:

$$
{\Huge
\alpha_2(t) =
\frac{3\left\langle |\Delta \mathbf r(t)|^4 \right\rangle}
{5\left\langle |\Delta \mathbf r(t)|^2 \right\rangle^2}
-1
}
$$

The anisotropic workflow writes directional pair parameters:

$$
{\Huge
\alpha_{ij}(t) =
\frac{\left\langle \Delta r_i^2(t)\Delta r_j^2(t) \right\rangle}
{\left\langle \Delta r_i^2(t) \right\rangle
 \left\langle \Delta r_j^2(t) \right\rangle}
-1
\quad
(ij=xy,xz,yz)
}
$$

and the averaged anisotropy parameter:

$$
{\Huge
\alpha_{\mathrm{ani}}(t) =
\frac{\alpha_{xy}(t)+\alpha_{xz}(t)+\alpha_{yz}(t)}{3}
}
$$

These quantities are dimensionless; MSD has units of length squared.

## Correlation Functions

Correlation-function workflows are described in [Module 4: Correlation Functions](Manual.md#module-4-correlation-functions), [Correlation Function Parameters](Manual.md#correlation-function-parameters), and [Correlation Normalization](Manual.md#correlation-normalization).

For two scalar or vector observables \(A\) and \(B\), the unnormalized correlation has the form:

$$
{\Huge
C_{AB}(t) =
\left\langle A(t_0)\,B(t_0+t) \right\rangle_{t_0}
}
$$

or, for vector data:

$$
{\Huge
C_{\mathbf A\mathbf B}(t) =
\left\langle \mathbf A(t_0)\cdot\mathbf B(t_0+t) \right\rangle_{t_0}
}
$$

When mean subtraction is enabled:

$$
{\Huge
\delta A(t) = A(t)-\langle A\rangle
}
$$

and the same correlation is formed from the fluctuations. The saved correlation curve is normalized as:

$$
{\Huge
C_{\mathrm{out}}(t)=\frac{C(t)}{C(0)}
}
$$

The separate variance output stores \(C(0)\) after applying the user coefficient.

## Diffusion Constants

The diffusion module is described in [Module 5: Diffusion Constant](Manual.md#module-5-diffusion-constant), [D VACF](Manual.md#d-vacf), [D MSD](Manual.md#d-msd), and [Infinite Size Correction Parameters](Manual.md#infinite-size-correction-parameters).

The VACF route uses the Green-Kubo relation:

$$
{\Huge
D_{\mathrm{VACF}} =
\frac{1}{3}\int_0^\infty
\left\langle \mathbf v(0)\cdot\mathbf v(t) \right\rangle\,dt
}
$$

The velocity relaxation time is the integral of the normalized VACF:

$$
{\Huge
\tau_v =
\int_0^\infty
\frac{\left\langle \mathbf v(0)\cdot\mathbf v(t) \right\rangle}
{\left\langle \mathbf v(0)\cdot\mathbf v(0) \right\rangle}\,dt
}
$$

If the VACF input is already normalized, the user-provided velocity variance supplies the missing scale:

$$
{\Huge
D_{\mathrm{VACF}} =
\frac{\left\langle v^2 \right\rangle}{3}\tau_v
}
$$

The MSD route fits the long-time relation:

$$
{\Huge
\mathrm{MSD}(t) \approx a t
}
$$

and reports:

$$
{\Huge
D_{\mathrm{MSD}}=\frac{a}{6}
}
$$

Equivalently:

$$
{\Huge
D_{\mathrm{MSD}} =
\lim_{t\to\infty}
\frac{\left\langle |\Delta \mathbf r(t)|^2 \right\rangle}{6t}
}
$$

When enabled, the finite-size correction is applied after the raw diffusion estimate; the required GUI fields and output records are listed in [Infinite Size Correction Parameters](Manual.md#infinite-size-correction-parameters) and [Infinite Size Correction Output](Manual.md#infinite-size-correction-output).

## Radial Distribution Functions

The RDF workflow is described in [Extra Module: Radial Distribution Function](Manual.md#extra-module-radial-distribution-function), [RDF Fields](Manual.md#rdf-fields), and [RDF Output](Manual.md#rdf-output).

For two zero-based selections \(A\) and \(B\), the software accumulates minimum-image pair distances between particles in `Selection 1` and particles in `Selection 2`. Empty selections mean all particles. The selected particle counts are:

$$
{\Huge
N_A=|A|,
\quad
N_B=|B|
}
$$

For a frame \(t\), each selected pair distance is computed from:

$$
{\Huge
\Delta \mathbf r_{ab}(t)=
\mathbf r_a(t)-\mathbf r_b(t)
-\mathbf L\,\mathrm{round}\!\left(
\frac{\mathbf r_a(t)-\mathbf r_b(t)}{\mathbf L}
\right),
\quad
d_{ab}(t)=|\Delta \mathbf r_{ab}(t)|
}
$$

where \(a\in A\), \(b\in B\), and \(\mathbf L=(L_x,L_y,L_z)\). With `Exclude Self-Pairs` enabled, pairs with the same zero-based particle index are omitted:

$$
{\Huge
\mathcal P =
\left\{(a,b): a\in A,\ b\in B,\ a\ne b\right\}
}
$$

Without `Exclude Self-Pairs`, the pair set is simply \(A\times B\). For bin edges \(r_i=i\Delta r\), the reported radius is the bin center:

$$
{\Huge
r_i^{\mathrm{center}}=
\frac{r_i+r_{i+1}}{2}
}
$$

The raw histogram count for bin \(i\) is:

$$
{\Huge
H_i=
\sum_{m}
\sum_{t\in m}
\sum_{(a,b)\in\mathcal P}
\mathbf 1\!\left[
r_i \le d_{ab}(t) < r_{i+1}
\right]
}
$$

where \(m\) runs over the selected coordinate files. This is why split trajectory pieces are accumulated into one total RDF instead of being normalized separately.

The shell volume for a radial bin \([r_i,r_{i+1})\) is:

$$
{\Huge
\Delta V_i =
\frac{4\pi}{3}\left(r_{i+1}^3-r_i^3\right)
}
$$

`Density-Normalize` controls whether the Selection 2 number density is included in the denominator:

$$
{\Huge
f_{\rho}=
\begin{cases}
\rho_B, & \text{Density-Normalize checked}\\
1, & \text{Density-Normalize unchecked}
\end{cases}
}
$$

The radial profile reported in column 2 is then:

$$
{\Huge
y(r_i^{\mathrm{center}})=
\frac{H_i}
{N_{\mathrm{frames}}\,N_A\,f_{\rho}\,\Delta V_i}
}
$$

where:

$$
{\Huge
N_{\mathrm{frames}}=
\sum_m F_m,
\quad
V=L_xL_yL_z,
\quad
\rho_B=\frac{N_B}{V}
}
$$

Here \(F_m\) is the number of retained frames from coordinate file \(m\) after trajectory selection, stride, and optional desired-length filtering. When `Density-Normalize` is checked, \(y(r)=g(r)\), the usual dimensionless RDF. When it is unchecked, \(y(r)\) is a shell-volume-normalized radial number-density profile around Selection 1, with units of inverse volume.

The running coordination number is:

$$
{\Huge
N(r_n^{\mathrm{center}})=
\sum_{i=0}^{n} y(r_i^{\mathrm{center}})\,f_{\rho}\,\Delta V_i
}
$$

The output also includes `hist`, which is \(H_i\), the raw pair-count histogram before RDF normalization. `Chunkify`, `Chunk Size 1`, and `Chunk Size 2` only split the pair-distance work into smaller blocks for memory control; they do not change these equations.

## Density Structure Factors

Density structure factors are described in [Module 6: Static Structure Factor](Manual.md#module-6-static-structure-factor), [Density Structure Factor Tab](Manual.md#density-structure-factor-tab), [Isotropic](Manual.md#isotropic), [Directional](Manual.md#directional), and [Along k Components](Manual.md#along-k-components).

For explicit reciprocal vectors, the density amplitude is:

$$
{\Huge
\rho(\mathbf k) =
\sum_{j=1}^{N}\exp\!\left(i\mathbf k\cdot\mathbf r_j\right)
}
$$

and the directional density structure factor is:

$$
{\Huge
S(\mathbf k) =
\frac{1}{N}\left|\rho(\mathbf k)\right|^2
}
$$

For isotropic density calculations, the pair-distance expression is:

$$
{\Huge
S(k) =
1+\frac{2}{N}
\sum_{i=1}^{N-1}\sum_{j=i+1}^{N}
\frac{\sin(k r_{ij})}{k r_{ij}}
}
$$

The reciprocal vector components are generated from the orthorhombic cell:

$$
{\Huge
k_x=\frac{2\pi n_x}{L_x},
\quad
k_y=\frac{2\pi n_y}{L_y},
\quad
k_z=\frac{2\pi n_z}{L_z}
}
$$

with the all-zero vector excluded. The three `k` tiers, shell width, cutoffs, and component restrictions are documented in [What The Three k Tiers Mean](Manual.md#what-the-three-k-tiers-mean), [What k_x, k_y, and k_z Mean](Manual.md#what-k_x-k_y-and-k_z-mean), and [Along k Components](Manual.md#along-k-components).

## Charge-Dipole Structure Factors

Charge-dipole structure factors are described in [Charge-Dipole Structure Factor Tab](Manual.md#charge-dipole-structure-factor-tab), [Charge-Dipole Fields](Manual.md#charge-dipole-fields), [Charge-Dipole k Parameters](Manual.md#charge-dipole-k-parameters), [Charge-Dipole Directional Mode](Manual.md#charge-dipole-directional-mode), [Charge-Dipole Isotropic Mode](Manual.md#charge-dipole-isotropic-mode), and [Charge-Dipole Cutoffs And Extra Output](Manual.md#charge-dipole-cutoffs-and-extra-output).

The charge-dipole displacement vector is:

$$
{\Huge
\mathbf r_{q,p}=\mathbf r_p-\mathbf r_q
}
$$

It points from the charge site \(q\) to the dipole position \(p\). The unit vector is:

$$
{\Huge
\hat{\mathbf r}_{q,p} =
\frac{\mathbf r_{q,p}}{|\mathbf r_{q,p}|}
}
$$

Only nonzero charge sites contribute. With real-space cutoffs, only dipoles within the cutoff distance of those nonzero charge sites are included.

For isotropic mode with `Small k approx` unchecked:

$$
{\Huge
S_{qp}(k) =
-\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\hat{\mathbf r}_{q,p}\right)
j_1\!\left(k|\mathbf r_{q,p}|\right)
}
$$

where \(j_1(x)\) is the spherical Bessel function of the first kind:

$$
{\Huge
j_1(x)=\frac{\sin x}{x^2}-\frac{\cos x}{x}
}
$$

For isotropic mode with `Small k approx` checked:

$$
{\Huge
S_{qp}(k) \simeq
-\frac{k}{3}
\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\mathbf r_{q,p}\right)
}
$$

For directional mode with `Small k approx` unchecked:

$$
{\Huge
S_{qp}(\mathbf k) =
i\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\hat{\mathbf k}\right)
\exp\!\left(i\mathbf k\cdot\mathbf r_{q,p}\right)
}
$$

For directional mode with `Small k approx` checked:

$$
{\Huge
S_{qp}(\mathbf k) \simeq
-k\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\hat{\mathbf k}\right)
\left(\hat{\mathbf k}\cdot\mathbf r_{q,p}\right)
}
$$

The directional small-`k` approximation is real-valued. The output keeps the directional real/imaginary column layout described in [Charge-Dipole Directional Mode](Manual.md#charge-dipole-directional-mode), with the imaginary column equal to zero in small-`k` mode.

The file-based charge-dipole workflows report raw accumulated `Sqp(k)` values. They are not normalized by the number of frames, trajectories, or dipoles, matching the mode descriptions in [Charge-Dipole Isotropic Mode](Manual.md#charge-dipole-isotropic-mode) and [Charge-Dipole Directional Mode](Manual.md#charge-dipole-directional-mode).

## Output and Workflow Notes

Generated scripts may also write per-trajectory reports, status logs, and cutoff dipole-count files depending on the selected module. Static-structure-factor auxiliary outputs are listed in [Density-Tab Auxiliary Files](Manual.md#density-tab-auxiliary-files) and [Charge-Dipole Auxiliary Files](Manual.md#charge-dipole-auxiliary-files). The GUI remains a workflow generator: the expensive calculations run later when the generated scripts are executed; see [Final notes](Manual.md#final-notes).
