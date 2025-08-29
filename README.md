# CUAVA-1 Eccentricity and Space Weather Analysis

This project explores associations between the orbital eccentricity of the **CUAVA-1 CubeSat** and space weather activity.
The analysis combines CUAVA-1 orbital data (TLEs) with space weather index logs during CUAVA-1's mission to study periodicities, correlations, and potential lags between orbital dynamics and solar–geomagnetic conditions. I conducted this research under the supervision of Prof. Iver Cairns (USYD) as part of the Physics Special Studies Program.

## Data Sources
- **Satellite orbital data**: TLEs from [CelesTrak / US Space Force](https://celestrak.org).  
- **Space weather indices**: F10.7cm solar flux, Kp, Dst, and Ap from [OMNIWeb Plus / NASA](https://omniweb.gsfc.nasa.gov/).

## Methods
The script implements:
- Parsing and preprocessing of TLE eccentricity data.  
- Parsing OMNI space weather datasets.  
- Time derivative of eccentricity (de/dt).  
- Smoothing via Savitzky–Golay filters.  
- Lomb–Scargle Periodograms to identify periodicities in both orbital and space weather time series.  
- Cross-correlation analysis to evaluate lag times between eccentricity/eccentricity-derivative and space weather indices.  
- Estimation of uncertainties (FWHM on spectral peaks and lag correlations).

## Folder Structure

- `src/`: `CUAVA1eccentricityandspaceweather_analysis.py` — main analysis script.  
- `data/`: TLE and space weather indices data

## Requirements
Python 3.10+ with the following packages:
- numpy  
- matplotlib  
- pandas  
- scipy  
- astropy  
- statsmodels  

Install with:
```bash
pip install numpy matplotlib pandas scipy astropy statsmodels
```

## Usage
1. Place the TLE file (`cuava1.txt`) and OMNI space weather file (`CUAVA-1_Omni2.txt`) in the repository.
2. Run the main script:
```bash
python CUAVA1eccentricityandspaceweather_analysis.py
```
3. The script will produce figures showing:
   - Eccentricity derivatives alongside solar flux.  
   - Periodograms comparing orbital and space weather frequencies.  
   - Cross-correlation plots with lag/uncertainty estimates.

## Author
Developed by **Yunki Yau**  
GitHub: [yunkiyau](https://github.com/yunkiyau)  
Email: yunki.yau@gmail.com, yyau2516@uni.sydney.edu.au


## Acknowledgements
Supervised by Prof. Iver Cairns - USYD Professor of Space Physics, Director, CUAVA: the ARC Training Centre for CubeSats, UAVs, and Their Applications.


