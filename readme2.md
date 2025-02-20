# Drivence

This repository provides the code of the paper "**Drivence: Realistic Driving Sequence Synthesis for Testing
Multi-sensor Fusion Perception Systems**"

[[website]](https://sites.google.com/view/drivence)

![](https://github.com/853108389/AI-MSF-drivence_test/blob/master/src/0_110_pc_labels_gen_virmot.gif)
![](https://github.com/853108389/AI-MSF-drivence_test/blob/master/src/0_110_pic_labels_gen_virmot.gif)


Drivence is a realistic driving sequence synthesis tool for testing multi-sensor fusion (MSF) perception systems. It can
generate realistic driving sequences with various scenarios. Drivence can be used to evaluate the robustness of MSF
perception systems under different traffic conditions. It involves the following steps:

[](https://github.com/853108389/drivence_test/blob/master/src/workflow.png)

**Occupancy Grid Mapping**.
Drivence introduces an occupancy grid mapping module that generates an occupancy grid map from the input point cloud
data of the driving sequence

**Trajectory Generation**.
Drivence introduces a three-step trajectory generation process. First, we use a clustering method to identify waypoints
representing all possible driving lanes within the drivable area. Next, a global path planner is employed to generate a
global reference trajectory that aligns with the identified lanes and follows a specified driving pattern. Finally, we
utilize a local path planner to dynamically adjust this reference trajectory in real time to avoid collisions with
existing NPC cars.

**Physical-Aware Multi-Sensor Simulation**
Drivence leverage a physical-aware multi-sensor simulation module to insert multiple types of objects into the driving
sequence. The module can simulate the sensor data of various sensors, including LiDAR and camera, to generate realistic
sensor data for testing MSF perception systems.

**Metamorphic Testing**
Drivence leverage metamorphic relations (MRs) to develop test oracles that automatically assess the system’s
performance.

## News
We plan to prepare a demo code for quick start. Please stay tuned.

## Installation

We implement Drivence with PyTorch 1.8.0 and Python 3.7.11. All experiments are conducted on a server with an Intel
i7-10700K CPU (3.80 GHz), 48 GB RAM, and an NVIDIA GeForce RTX 3070 GPU (8 GB VRAM).

```
pip install -r requirements.txt
```

### Install MSF-based Systems

In order to reproduce our experiments, we need to carefully configure the environment for each system.
For DFmot and JMODT, we can refer the benchmarking code in
the [`AI-MSF-Benchmark` repository](https://sites.google.com/view/ai-msf-benchmark/benchmark)
The YONTD follow the instructions in the [YONTD repository]() and VirTrack follow the instructions in
the ![VirTrack repository]()

## Visualizations and Case Studies

We prepare many visualization results.
Please refer to our [website](https://sites.google.com/view/drivence/experiment/faults-visualization) for
more [visualization results](https://sites.google.com/view/drivence/data-visualization).

## Usage

1. Run map generation
    ```
    cd map_gen
    python main.py
    ```
   
2. Run trajectory generation
    ```
    cd TrackGen
    python main.py
     ```
   
3. Run sensor simulation 
   ```
      cd data_gen
      python main_function_new.py
   ```
   We can run main.py to generate the driving sequence and sensor data.

4. testing 
   ```
      python evaluate_script.py
   ```
   This script includes the all system's testing process. The results are stored in each system's output folder.

5. results statistics
   ```
      python statistic_faults.py
      python statistic_hota.py
   ```


## Citation

Please cite the following paper if `Drivence` helps you on the research:

```

```

## Note that

We will update the code with the paper updates. Please stay tuned.