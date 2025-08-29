# CUAVA-1 Eccentricity and Space Weather Analysis

This project explores associations between the orbital eccentricity of the **CUAVA-1 CubeSat** and space weather activity.  
The analysis combines satellite orbital data (TLEs) with space weather indices to study periodicities, correlations, and potential lags between orbital dynamics and solar–geomagnetic conditions.

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

## Files
- `CUAVA1eccentricityandspaceweather_analysis.py` — main analysis script.  
- `cuava1.txt` — TLE data file (to be added).  
- `CUAVA-1_Omni2.txt` — space weather index data file (to be added).

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
Email: yunki.yau@gmail.com  

---
