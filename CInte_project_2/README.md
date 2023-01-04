# Computational Intelligence project
Solution for TSP using single objective genetic algorithm.

## Files:
SO_utils.py includes :
- evaluation function, intialization of population using with and without use of heuristic solution,
- additional functions to plot path, convergence_curve.

TSP_SO.py includes :
- file with all parameters,
- function run() contain main evolving algorithm with creation of population, selection and mutation,
- main function runs run() function 30 times on the same seeds in order to produce same results and obtain best possible solution by calucalation of all means and standard deviations.
    
## Example Results
Convergence curve:

![Figure 2021-11-14 163924](https://user-images.githubusercontent.com/32979274/141690052-3839a525-a897-44dc-a746-3e1828b9d10d.png)

Best solution:

![Figure 2021-11-14 163919](https://user-images.githubusercontent.com/32979274/141690054-dcd8070e-0f54-4c49-8655-cefb2a41a712.png)
