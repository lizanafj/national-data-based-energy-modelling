# A national data-based energy modelling to support heating electrification 

This python code contains a national data-based energy modelling to identify optimal heat storage capacity to support heating electrification.

This approach is further detailed in the following scientific article: 

  **J. Lizana, et al. (2022) A national data-based energy modelling to identify optimal heat storage capacity to support heating electrification. Energy. https://doi.org/10.1016/j.energy.2022.125298**

## Overview

Novel modelling approach that utilises easily accessible national-level data to identify the required heat storage volume in domestic heating systems to decrease peak power demand and maximises carbon reductions associated with electrified heating technologies through smart demand-side response. The approach assesses the optimal shifting of heat pump operation to meet thermal heating demand according to different heat storage capacities, which are defined in relation to the time (in hours) in which the heating demand can be provided directly from the heat battery, without heat pump operation. 

Input data per country are obtained from four different open access databases: 

	-Eurostat open dataset (A)

	-International Energy Agency (B) 

	-Data Platform: When2Heat (C) 

	-ElectricityMap database (D)

The approach is divided into five steps: 

	-heating demand (1)

	-demand scenarios (2)

	-grid scenario (3)

	-electricity load (4)

	-GHG emissions (5)

A detailed description of data and methods are included in the article.
This code is an example of the workflow, using data from Spain and the UK in 2018. 

