- Temperature is now measured as $\tilde T = k_B T / J = 1/\beta J$, whereas in previous implementations it was measured as $\tilde T = \beta J$. I made this change as $\tilde T \propto T$ which feels more intuitive to me. This also means that the extra hyperparameter $\beta\alpha$ is now $\alpha/J$.
- In the current implementation. the computation of structure factor is hard-coded to remove the central pixel, which is simply total magnetisation. This can be undone in the next stage.
- Current implementation is, for each digit:
  - Run a single simulation until equilibrium (10k steps)
  - Measure structure factor every 10 sweeps for a total of 100 times
  - Store average structure factor
  It may be better to run a separate simulation for each measurement of structure factor, but presently converting the whole dataset takes 5-6 hours. This improvement can be implemented at a later date.