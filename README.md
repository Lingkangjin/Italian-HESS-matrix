# Hybrid Energy Storage System (HESS)
The objective of the work is to assess the optimal design of PV-BESS-Electrolyser.<br>

Driven by the fact that the obbjective function is to keep the hydrogen  production stable as possible.<br>
Althought from aggregate point of view, it is "easy", what we need is to have an energy balance, at least hourly based ones.
<br>
The systems configuration is divided into three parts: <br>
1. PV panels
2. BESS    
3. Electrolyser

## PV panels:
Is the main input of the model, and it represents the produced energy, the main point is scale everything using the PV capacity;
<br>
Therfore, for sake of simplicity, is taken as **1 kWp**, and its hourly production is assessed using a webtool from JRC:
https://re.jrc.ec.europa.eu/pvg_tools/en/ 

## BESS
The battery energy storage systems, functions as buffer to supply the electrolyser when the PV panel cannot supply, i.e. fluctuations

## Water electrolyser
the core of the model, and the objective is to have the most constant production of the electrolysis

## Optimisation model eqautions
The optimisation model is constrcuted using the Pyomo library, solved using Gurobi solver.

### parameters
$\eta_{EZ}$=electrolyser efficiency=0.7
