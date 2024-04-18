# vrp-challenge
Coding a solutions to the Vehicle Routing Problem (VRP) for Vorto interview process

## Execution
### standalone
> python mySubmission.py <filename>

### evaluation script
> python evaluateShared.py --cmd "python mySubmission.py" --problemDir training_problems

## first thoughts on how to approach this before I dig into online resources
- load data from path/file ... what does the data look like
- use numpy or pandas to compute all distances up front (o->p, p->d, d->o)?
- organize all points in graph form with distances between each poi -- part 2, let's start with heuristic approach
- add a driver and get the closest pickup
    - at dropoff, find nearest pickup (pure NN)
    - can driver -> pu -> do -> o in time?
        - Y: do it and remove job from queue
        - N: check other routes (NN order) ... no routes add origin cost and retire
- more routes in queue?  Add driver
- repeat above until queue is empty

## to optimize
- run multiple times with different seed route?
- drivers head in opposite directions to avoid overlap
- what happens with the simplest models?
  - 1 driver, all loads
  - n drivers, one load
- is all data randomized?  Or are there patterns to exploit
- implement graph solution to clean up heuristic approach
- reduce data type checking
- figure out why many drivers only run one route (likely reduce greediness in heuristic)

## key known points
- number of drivers is unbounded
- Driver can only drive for 12 hours a day
- Drivers "cost", so one load per driver is going to be VERY expensive 500 minutes
- Driver goes from (x, y) to (x1, y1)
- Time is calculated as the euclidean distance between two points
  - Time = sqrt((x1 - x)^2 + (y1 -e = 141.42 minutes
- Driday ver starts and ends at (0,0)
- Total cost = 500 * number of drivers + total # of driven minutes
- Low total cost with good efficiency is the goal
- Do not have to determine feasibility ... all problems are solvable
- Max of 200 loads

## cleaning things up
* giving up on using a queue & recalibrating NN after removing jobs ... too many places where I'm fighting index pointers
* adding the decorator to clean up some assert stuff has slowed us down slightly
    * was showing 517 mS through evaluation script after rebuild
    * now showing 608 mS
    * ... disable decorators for submission :( ... back to 515 mS

## research sources
- https://vrpy.readthedocs.io/en/latest/index.html ... python library for VRP problems
- https://medium.com/@writingforara/solving-vehicle-routing-problems-with-python-heuristics-algorithm-2cc57fe7079c
- OpenAI. "GPT-4 Model." OpenAI, 2023, www.openai.com.