# Graphical User Interface for pyPCGA

This project focuses on developing a Graphical User Interface for professionals to use pyPCGA as a tool for decision making in the field along with students and educators who study data science in hydrogeology.


# Modules 

* **Module 1:** Takes parameters to define the x,y,z dimensions from min to max possible values. Outputs M, the number of unknown values.
* **Module 2:** Works with the kernel to pass permissions to PCGA algorithm to run groundwater model executable.
* **Module 3:** Takes file with observation data to be used by PCGA. Calculates and verifies correctness of N.
* **Module 3:** Takes inversion parameters to use during iterations of the PCGA optimization algorithm. 

# User Experience Flowchart

<p align="center">
  <img src="/info/pyPCGA_Flowchart.svg" width="546" height="863" >
</p>

# Version

* _*v1.0.0*_
  * - [x] Integrated PyQt with Pumping Identification's case study source code
  * - [x] Input for Module 1, 2, 3, and 4 
  * - [x] Choose file, import input, and pass it to an array
  * - [x] Values & visualization output
  * - [x] Basic error handling
  * - [x] Access kernel to pass permissions to PCGA algorithm to run groundwater model executable
  * - [x] Execution and Restart buttons functional
  * - [x] Calculation of final values with second visualization

* _*v1.0.1*_
  * - [ ] Change of label names in Module 1
  * - [ ] Adding a conditional to autofill Y and X axis values
  * - [ ] Set an optional for s_init to choose a constant or file import
  * - [ ] Naming for M and N in both backend and frontend
  * - [ ] Updated error handling


# Example Notebooks

* [1D linear inversion example](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/pumping_history_identification/linear_inverse_problem_pumping_history_identification.ipynb) (from Stanford 362G course)

* [1D nonlinear inversion example](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/pumping_history_identification/nonlinear_inverse_problem_pumping_history_identification.ipynb) (from Stanford 362G course)

* [Hydraulic conductivity estimation example using USGS-FloPy (MODFLOW)](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/modflow_flopy/inversion_modflow.ipynb) [Lee and Kitanidis, 2014]

* [Tracer tomography example using Crunch](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/tracer_tomography_ade_crunch/inversion_example_advection_diffusion_crunchtope.ipynb) (with Mahta Ansari from UIUC Druhan Lab)

* [Bathymetry estimation example using STWAVE](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/stwave_duck/inversion_stwave.ipynb) (with USACE-ERDC-CHL)

* [Permeability estimation example using TOUGH2](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/tough_heat/joint_inversion_example_tough.ipynb) (with Amalia Kokkianki, USFCA)

* MODFLOW-USG/SEAWAT/MODFLOW6/E4D/ADH examples coming soon! 

# Credits

pyPCGA is based on Lee et al. [2016] and currently used for Stanford-USACE ERDC project led by EF Darve and PK Kitanidis and NSF EPSCoR `Ike Wai project. 

Code contributors include:

* Jonghyun Harry Lee 
* Matthew Farthing
* Ty Hesser (STWAVE example)

FFT-based matvec code is adapted from Arvind Saibaba's work (https://github.com/arvindks/kle). 

# References

- J Lee, H Yoon, PK Kitanidis, CJ Werth, AJ Valocchi, "Scalable subsurface inverse modeling of huge data sets with an application to tracer concentration breakthrough data from magnetic resonance imaging", Water Resources Research 52 (7), 5213-5231

- AK Saibaba, J Lee, PK Kitanidis, Randomized algorithms for generalized Hermitian eigenvalue problems with application to computing Karhunen–Loève expansion, Numerical Linear Algebra with Applications 23 (2), 314-339

- J Lee, PK Kitanidis, "Large‐scale hydraulic tomography and joint inversion of head and tracer data using the Principal Component Geostatistical Approach (PCGA)", WRR 50 (7), 5410-5427

- PK Kitanidis, J Lee, Principal Component Geostatistical Approach for large‐dimensional inverse problems, WRR 50 (7), 5428-5443

# Applications

- J Lee, H Ghorbanidehno, M Farthing, T. Hesser, EF Darve, and PK Kitanidis, Riverine bathymetry imaging with indirect observations, Water Resources Research, 54(5): 3704-3727, 2018

- J Lee, A Kokkinaki, PK Kitanidis, Fast large-scale joint inversion for deep aquifer characterization using pressure and heat tracer measurements, Transport in Porous Media, 123(3): 533-543, 2018

- PK Kang, J Lee, X Fu, S Lee, PK Kitanidis, J Ruben, Improved Characterization of Heterogeneous Permeability in Saline Aquifers from Transient Pressure Data during Freshwater Injection, Water Resources Research, 53(5): 4444-458, 2017

- S. Fakhreddine, J Lee, PK Kitanidis, S Fendorf, M Rolle, Imaging Geochemical Heterogeneities Using Inverse Reactive Transport Modeling: an Example Relevant for Characterizing Arsenic Mobilization and Distribution, Advances in Water Resources, 88: 186-197, 2016
