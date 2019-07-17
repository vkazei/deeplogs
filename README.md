# deeplogs
Velocity model building by deep learning. Multi-CMP gathers are mapped into velocity logs.

This repository reproduces the results of the paper: 

Kazei, V., Ovcharenko, O., Zhang, X., Peter, D. & Alkhalifah, T. 
**"Mapping seismic data cubes to vertical velocity profiles by deep learning: New full-waveform inversion paradigm?"**,
Geophysics, submitted (2019) 

Run:

    data/velocity_logs_from_seismic.ipynb

Common-midpoint gathers are used to build a velocity log at the central midpoint location. 
![cmp_to_log](latex/Fig/relevantCMP.png)
![cmp_to_log](latex/Fig/in_out_shape.png)

Velocity model is then retrieved as an assembly of depth profiles.




