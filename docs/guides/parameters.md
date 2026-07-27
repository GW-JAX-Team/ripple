# Parameters and Conventions

Every waveform's `params` dict uses a name from this glossary.
Consult a model's `parameter_names` for the ordered set a specific model expects — not every model uses every parameter below.

## Mass

| Name | Meaning | Units |
| --- | --- | --- |
| `M_c` | Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / (m_1+m_2)^{1/5}$ | solar masses |
| `eta` | Symmetric mass ratio $\eta = m_1 m_2 / (m_1+m_2)^2 \in (0, 0.25]$ | dimensionless |

ripple parameterises by $(\mathcal{M}, \eta)$ rather than component masses $(m_1, m_2)$ — use `ripplegw.conversions` to convert between them:

```python
import jax.numpy as jnp
from ripplegw.conversions import Mc_eta_to_ms, ms_to_Mc_eta

m1, m2 = Mc_eta_to_ms(jnp.array([Mc, eta]))    # -> component masses, m1 >= m2
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))    # -> chirp mass, symmetric mass ratio
```

!!! warning "Packed arrays, not separate scalars"
    Every `ripplegw.conversions` function takes a single packed array — `Mc_eta_to_ms(jnp.array([Mc, eta]))`, not `Mc_eta_to_ms(Mc, eta)`.

`ripplegw.conversions` and `ripplegw.constants` are not re-exported at the top level; import them explicitly.

## Spin

Aligned-spin models use only the spin components along the orbital angular momentum:

| Name | Meaning | Range |
| --- | --- | --- |
| `s1_z`, `s2_z` | Dimensionless aligned spin of the primary/secondary | $[-1, 1]$ |

Precessing models (`IMRPhenomPv2`, `IMRPhenomXP`, `IMRPhenomXPHM`) additionally take the in-plane components:

| Name | Meaning | Range |
| --- | --- | --- |
| `s1_x`, `s1_y`, `s2_x`, `s2_y` | In-plane dimensionless spin components | $[-1, 1]$ |

## Tidal deformability

Tidal models (`TaylorF2`, `IMRPhenomD_NRTidalv2`, `IMRPhenomXAS_NRTidalv3`) take the dimensionless tidal deformability of each body, in one of two parameterisations selected by the model's `use_lambda_tildes` constructor argument:

| `use_lambda_tildes` | Parameters | Meaning |
| --- | --- | --- |
| `False` (default) | `lambda_1`, `lambda_2` | Dimensionless tidal deformability of the primary/secondary |
| `True` | `lambda_tilde`, `delta_lambda_tilde` | The mass-weighted combinations of arXiv:1402.5156 Eq. 5–6 |

Convert between them with `ripplegw.conversions.lambdas_to_lambda_tildes` / `lambda_tildes_to_lambdas` (also packed-array signatures, `(lambda_1, lambda_2, mass_1, mass_2)`).

## Extrinsic parameters

| Name | Meaning | Units |
| --- | --- | --- |
| `d_L` | Luminosity distance | Mpc |
| `phase_c` | Coalescence phase | radians |
| `iota` | Inclination angle between the orbital angular momentum and the line of sight | radians |

`d_L` is what [`at_unit_distance`](waveforms.md#evaluating-at-a-fixed-distance) sets to 1.0.
There is no `t_c` (time of coalescence) parameter — see the [FAQ](../FAQ.md) for why.

## Burst parameters (`SineGaussian`)

| Name | Meaning | Units |
| --- | --- | --- |
| `Q` | Quality factor of the sine-Gaussian envelope | dimensionless |
| `f_0` | Central frequency | Hz |
| `hrss` | Root-sum-squared strain amplitude | dimensionless (strain) |
| `phase` | Phase | radians |
| `e` | Eccentricity — controls the relative $h_+$/$h_\times$ amplitude | $[0, 1]$ |

`SineGaussian`'s axis is a **time** grid centred at zero, not a frequency grid: `t = jnp.arange(-duration/2, duration/2, 1/fs)`.

## Physical constants

`ripplegw.constants` exposes the constants used internally (`MSUN`, `MRSUN`, `MTSUN`, `G`, `C`, `PI`, `TWO_PI`, `MPC`, `EULERGAMMA`) as plain Python floats, for anyone building parameter conversions of their own.
