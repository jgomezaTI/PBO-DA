# Implementation references

- Olivier Roussel, *Restricted OPB Format in Use in the PB Competitions*, version
  dated March 11, 2024:
  <https://www.cril.univ-artois.fr/PB24/OPBcompetition.pdf>
- Pseudo-Boolean Competition 2026:
  <https://www.cril.univ-artois.fr/PB26/>

The initial parser implements only the restricted linear section. Sources from Bryan
Alvarado's work and the experimental dataset should be added here once the exact
artifacts and licensing conditions are available.

## Upstream research software and benchmark definition

- Bryan Alvarado-Ulloa, *Backbone-based Predict and Search for Pseudo-Boolean
  Optimization*, UTFSM master's thesis, 2026:
  <https://cris.usm.cl/entities/tesis/e3f32d62-3f6e-41af-b189-b77db43c62ce>
- BackPaS source code: <https://github.com/bryan-alvarado-ulloa/backpas>
- GuroBack source code: <https://github.com/bryan-alvarado-ulloa/guroback>
- Huang et al., *Contrastive Predict-and-Search for Mixed Integer Linear Programs*,
  ICML 2024: <https://proceedings.mlr.press/v235/huang24w.html>

The thesis is the authoritative source for the BackPaS experimental scope used here:
MIS, MVC, and CA are the three binary PBO benchmarks; Item Placement is explicitly
excluded because it contains non-binary variables.
