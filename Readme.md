# Drivence

This repository provides the code for the paper:

**Drivence: Realistic Driving Sequence Synthesis for Testing Multi-Sensor Fusion Perception Systems**

[[Website]](https://sites.google.com/view/drivence)

[//]: # (![]&#40;https://github.com/853108389/drivence_test/blob/main/src/0_110_pic_labels_gen_virmot.gif&#41;  )

[//]: # (![]&#40;https://github.com/853108389/drivence_test/blob/main/src/0_110_pc_labels_gen_virmot.gif&#41;)

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/853108389/drivence_test/blob/main/src/0_110_pic_labels_gen_virmot.gif" alt="GIF 1" width="48%">
  <img src="https://github.com/853108389/drivence_test/blob/main/src/0_110_pc_labels_gen_virmot.gif" alt="GIF 2" width="48%">
</div>


Drivence is a realistic driving sequence synthesis tool designed for testing multi-sensor fusion (MSF) perception systems. It can generate realistic driving sequences under various scenarios and evaluate the robustness of MSF perception systems in different traffic conditions. The framework consists of the following components:

[](https://github.com/853108389/drivence_test/blob/main/src/workflow.png)

### **Occupancy Grid Mapping**  
Drivence introduces an occupancy grid mapping module that generates an occupancy grid map from input point cloud data of the driving sequence.

### **Trajectory Generation**  
Drivence features a three-step trajectory generation process:  
1. A clustering method identifies waypoints representing all possible driving lanes within the drivable area.  
2. A global path planner generates a global reference trajectory that aligns with the identified lanes and follows a specified driving pattern.  
3. A local path planner dynamically adjusts this reference trajectory in real-time to avoid collisions with existing NPC vehicles.

### **Physics-Aware Multi-Sensor Simulation**  
Drivence leverages a physics-aware multi-sensor simulation module to insert multiple types of objects into the driving sequence. This module can simulate data from various sensors, including LiDAR and cameras, to generate realistic sensor data for testing MSF perception systems.

### **Metamorphic Testing**  
Drivence utilizes metamorphic relations (MRs) to develop test oracles that automatically assess the system’s performance.

---

## **News**  
Considering the complexity of configuring the entire project, we are preparing a demo code for a quick start. Please stay tuned!

---

## **Installation**

Drivence is implemented using PyTorch 1.8.0 and Python 3.7.11. All experiments are conducted on a server equipped with an Intel i7-10700K CPU (3.80 GHz), 48 GB RAM, and an NVIDIA GeForce RTX 3070 GPU (8 GB VRAM).

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### **Install MSF-Based Systems**

To reproduce our experiments, the environment must be carefully configured for each system:  
- For **DFmot** and **JMODT**, refer to the benchmarking code in the [`AI-MSF-Benchmark` repository](https://sites.google.com/view/ai-msf-benchmark/benchmark).  
- For **YONTD**, follow the instructions in the [YONTD repository]().  
- For **VirTrack**, follow the instructions in the ![VirTrack repository]().

---

## **Visualizations and Case Studies**  

We provide extensive visualization results. Please refer to our [website](https://sites.google.com/view/drivence/experiment/faults-visualization) for more [visualization results](https://sites.google.com/view/drivence/data-visualization).

---

## **Usage**

1. **Run map generation**  
    ```bash
    cd map_gen
    python main.py
    ```

2. **Run trajectory generation**  
    ```bash
    cd TrackGen
    python main.py
    ```

3. **Run sensor simulation**  
    ```bash
    cd data_gen
    python main_function_new.py
    ```
    Run `main.py` to generate the driving sequence and sensor data.

4. **Run testing**  
    ```bash
    python evaluate_script.py
    ```
    This script includes the testing process for all systems. The results are stored in each system's output folder.

5. **Results statistics**  
    ```bash
    python statistic_faults.py
    python statistic_hota.py
    ```

---

## **Citation**

If **Drivence** helps your research, please cite the following paper:

```bibtex

```

---

## **Note**

We will update the code as the paper evolves. Please stay tuned!
