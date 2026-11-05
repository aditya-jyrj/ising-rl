- In the current implementation. the computation of structure factor is hard-coded to remove the central pixel, which is simply total magnetisation. This can be undone in the next stage.
- Current implementation is, for each digit:
  - Run a single simulation until equilibrium (10k steps)
  - Measure structure factor every 10 sweeps for a total of 100 times
  - Store average structure factor
  It may be better to run a separate simulation for each measurement of structure factor, but presently converting the whole dataset takes 5-6 hours. This improvement can be implemented at a later date.

  Data files:
  - `10000_100_10.npz`: 10000 sweeps, 100 structure factor measurements with intervals of 10 sweeps
  - `5000_100_50.npz`: 5000 sweeps, 100 structure factor measurements with intervals of 50 sweeps


- fix $\beta\alpha=1$ and scan different $\beta J$, eg from 0 to 10. take a look at the best range to choose.
  - can scan in a log scale : 0.1, 0.2, 0.4, 0.8, etc first, and then afterwards zoom in for more fine-scale tuning/measurement.
  
- observe performance change, if any
- we are observing how interactions affect the performance of machine learning
- machine exploits the property of the alignment of the spins
  - maybe look at how exactly the machine observes each image.
- how physics affects machine learning performance
- structure factor computation involves fourier transform, which discards rotational information (why does the structure factor image in `introduction.ipynb` not look symmetric then?)
  - symmetry and machine learning are related

- if we have the time, can try XY model.