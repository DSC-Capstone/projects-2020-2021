Scale model Group
The purpose of this project is to provide a testing ground for the transmission code in SchoolABM. By scaling down the parameters to a single room of students, effects of elements such as distance and ventilation can be seen more easily.
This also provides a testing environment that has a significantly faster runtime and has been useful in verifying our math for transmission both through droplets and aerosols

Test Simulation can be run using: python -m run 'test'

Visualization can be run using: python -m run 'visualize'

Visualization includes:
-Color viz indicating which students are infected
-Distribution plot of transmission rates post-processing (for each 5 minute step and unique infected individual)
-Time series plot of when infection occurs

TODO:
Docker Image (due thursday)
Airavata (due wednesday)

Well-Mixed room (due thursday)
