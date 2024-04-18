# vrp-challenge
Coding a solutions to the Vehicle Routing Problem (VRP) for Vorto interview process

# first thoughts on how to approach this before I dig into online resources
- load data from path/file ... what does the data look like?
- organize all points in graph form with distances between each point ???
- with calculated distances, can we quickly compute n-drivers?
- drivers head in opposite directions to avoid overlap
- what happens with the simplest models?
  - 1 driver, all loads
  - n drivers, one load each

# key known points
- number of drivers is unbounded
- Driver can only drive for 12 hours a day
- Drivers "cost" 500 minutes
- Driver goes from (x, y) to (x1, y1)
- Time is calculated as the euclidean distance between two points
  - Time = sqrt((x1 - x)^2 + (y1 - y)^2) minutes
  - i.e. (0, 0) -> (50, 50) -> (100, 100):
    - sqrt((50 - 0)^2 + (50 - 0)^2) = 70.71 minutes
    - sqrt((100 - 50)^2 + (100 - 50)^2) = 70.71 minutes
    - Total time = 141.42 minutes
- Driver starts and ends at (0,0)
- Total cost = 500 * number of drivers + total # of driven minutes
- Low total cost with good efficiency is the goal
- Do not have to determine feasibility ... all problems are solvable
- Max of 200 loads

# evaluation process


