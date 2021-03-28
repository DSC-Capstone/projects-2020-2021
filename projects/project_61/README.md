## Infection Risk App
In this project, We propose an application which estimates infection risk of COVID-19 in buildings. The application accepts building data and a set of parameters regarding occupants and infection rates of the surrounding community. Code and assumptions made in the algorithm are clearly explained to users for transparency, which those explainations are included in the project in our last quarter. 

Opening up the project
The source codes for the calculator are located in /src/calculator. To see the notebook containing the underlying logic and sample runnings for the calculator, open presentation in /notebooks. Note there are actual codes in those notebooks, because the purpose of this project is to create an infection estimation algorithm that's clear to users and easily understanable by users. It's important to show what the algorithm does for each steps. To see our UI demo, open Website in /notebooks. To visit our website, visit https://hinanawits.github.io/DSC-180B-Presentation-Website/ Notice we are constantly updating our website with newest data so if the Website notebooks and the website itself are inconsistant it means we updated something. 

To use run.py in command line, input python run.py [targets].

We currently have the following targets available:

test: which runs the calculator using sample parameters.
More information about those sample runs can be found in the report notebooks mentioned above.

Responsibilities

Etienne Doidic built the structure and underlying logic of the calculator, and also the notebooks for walk through.

Nicholas Kho helped developing the application and migrated the codes to src and project structure.

Zhexu Li added features to the application migrating the codes, updated the project structure and developed configs and run.py.
