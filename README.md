# ATLAS-Visualiser
Python visualisation tool that renders 3D particle tracks and energy deposits from simulated ATLAS detector collision events.

An interactive 3D particle physics visualisation framework built with **PyVista** and **VTK**, designed to ingest, process, and render real-world **ATLAS experiment ROOT data**.

## Preview

**Project Example Preview**

![Event](demos/Event.gif)


## Key Features

* **Real-Time Trajectory Simulation:** Computes Lorentz force helical trajectories using **NumPy** to accurately simulate charged particle paths under complex solenoidal magnetic fields.
* **Dynamic Event Animation Pipeline:** Features a robust three-phase animation engine that sequentially renders beam collisions, vertex interactions, outgoing lepton tracks, and Missing Transverse Energy (MET) vectors.
* **Optimised ROOT Data Ingestion:** Leverages **Uproot** for high-performance data extraction directly from High Energy Physics data trees, handling multi-lepton ($4l$) and missing energy event topologies.
* **Advanced Coordinate Transformations:** Automatically converts cylindrical and pseudorapidity coordinates into Cartesian space while performing real-time kinematic metric extraction.


## Tech Stack

* **Languages & Libraries:** Python, NumPy, PyVista, VTK, Uproot
* **Data Sources:** CERN ROOT (ATLAS Open Data / Experiment Trees)
* **Visualisation:** Interactive 3D rendering pipelines


## Overview of Architecture

1. **Ingestion (`Uproot`):** Pulls raw physics data trees from ROOT files, parsing kinematic variables and coordinate systems on the fly.
2. **Physics Engine (`NumPy`):** Applies relativistic and magnetic field calculations (Lorentz force) to generate realistic helical curves for charged particles.
3. **Render Pipeline (`PyVista`/`VTK`):** Assembles the collision event into an interactive 3D scene, orchestrating the multi-phase animation of tracks and energy vectors.