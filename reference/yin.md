
# YIN, a fundamental frequency estimator for speech and music

**Alain de Cheveigné**  
Ircam-CNRS, Paris, France  
**Hideki Kawahara**  
Wakayama University  

*(Received 7 June 2001; revised 10 October 2001; accepted 9 January 2002)*  

---

## Abstract  
An algorithm for estimating the fundamental frequency (\(F_0\)) of speech/musical sounds is presented. Based on autocorrelation with modifications to prevent errors, YIN achieves:  
- **3× lower error rates** vs. best competing methods (evaluated on laryngograph-verified speech)  
- **No upper limit** on frequency search range (suitable for high-pitched voices/music)  
- **Low latency** and minimal parameter tuning  
- **Extensible signal model** for handling aperiodicity  
- **Parallels with auditory processing** models  

---

## I. Introduction  
\(F_0\) (fundamental frequency) is the inverse of the period of a periodic signal. Challenges arise with real signals (speech/music) due to:  
- Departures from perfect periodicity (e.g., amplitude/frequency modulation)  
- Subjective pitch perception vs. physical \(F_0\) (e.g., pitch without periodicity)  

**Applications**:  
- Speech prosody/tonal languages, music transcription, multimedia indexing  
- Limitations: Reliability issues in existing methods  

**Contribution**: YIN algorithm combining autocorrelation with error-reduction steps.  

---

## II. The Method  
### Step 1: Autocorrelation Method  
Autocorrelation function (ACF) for discrete signal \(x_t\):  
\[
r_t(\tau) = \sum_{j=t+1}^{t+W} x_j x_{j+\tau} \quad \text{(modified ACF)} \tag{1}
\]  
*Problem*: Sensitivity to amplitude changes → "too low" errors.  

### Step 2: Difference Function  
Model: \(x_t - x_{t+T} = 0\) for period \(T\). Difference function:  
\[
d_t(\tau) = \sum_{j=1}^{W} (x_j - x_{j+\tau})^2 = r_t(0) + r_{t+\tau}(0) - 2r_t(\tau) \tag{6}
\]  
*Advantage*: Immune to amplitude changes → **Error rate ↓ 10.0% → 1.95%** (Table I).  

### Step 3: Cumulative Mean Normalized Difference  
\[
d_t'(\tau) = 
\begin{cases} 
1 & \tau = 0 \\
\frac{d_t(\tau)}{\frac{1}{\tau} \sum_{j=1}^{\tau} d_t(j)} & \tau > 0 
\end{cases} \tag{8}
\]  
*Benefits*:  
1. Eliminates zero-lag dip → **no upper frequency limit**  
2. Normalizes function → **Error rate ↓ 1.95% → 1.69%**  

### Step 4: Absolute Threshold  
- Set threshold (default: 0.1)  
- Select smallest \(\tau\) where \(d'(\tau) < \text{threshold}\)  
*Effect*: Reduces subharmonic errors → **Error rate ↓ 1.69% → 0.78%**  

### Step 5: Parabolic Interpolation  
Fit parabola to local minima of \(d'(\tau)\) for subsample resolution.  
*Impact*: Reduces fine errors at high \(F_0\).  

### Step 6: Best Local Estimate  
Search vicinity \([t - T_{\text{max}}/2, t + T_{\text{max}}/2]\) for minimum \(d'(\tau)\).  
*Result*: **Error rate ↓ 0.77% → 0.50%** (Table I).  

**Table I**: Gross error reduction across steps  
| Step | Error Rate (%) |
|------|----------------|
| 1    | 10.0           |
| 2    | 1.95           |
| 3    | 1.69           |
| 4    | 0.78           |
| 5    | 0.77           |
| 6    | 0.50           |

---

## III. Evaluation  
**Databases**: Laryngograph-verified speech (48 speakers, 1.9 hours).  
**Metric**: Gross error = \(|\text{est. } F_0 - \text{truth}| > 20\%\)  

**Results (Table II)**:  
- YIN outperforms 10+ methods across databases  
- **Avg. error: 1.03%** (vs. 3.1–16.8% for others)  
- 99% estimates accurate within 20%, 60% within 1%  

**Table II**: Gross error rates (%)  
| Method    | DB1  | DB2  | DB3  | DB4  | Avg.  |
|-----------|------|------|------|------|-------|
| pda       | 10.3 | 19.0 | 17.3 | 27.0 | 16.8  |
| ...       | ...  | ...  | ...  | ...  | ...   |
| **YIN**   | 0.30 | 1.4  | 2.0  | 1.3  | 1.03  |

---

## IV. Sensitivity to Parameters  
YIN is robust to parameter choices (Fig. 4):  
- **Window size**: Optimal at 25 ms  
- **Threshold**: Stable for 0.05–0.2  
- **Low-pass cutoff**: Best at ~1 kHz  

---

## V. Implementation Considerations  
- **Efficiency**: Recursive computation or FFT-based ACF  
- **Latency**: Can be reduced to \(T_{\text{max}} + T\)  
- **Confidence metric**: \(d'(T)\) indicates reliability  
- **Applications**: Real-time music tracking, MPEG7 metadata  

---

## VI. Extensions  
Extended models for handling aperiodicity:  
1. **Variable amplitude**:  
   \[
   d_t(\tau) = r_t(0) \left[1 - \frac{r_t(\tau)^2}{r_t(0) r_{t+\tau}(0)}\right] \tag{12}
   \]  
2. **Variable \(F_0\)**: Split window into segments  
3. **DC offset**:  
   \[
   d_t(\tau) = r_t(0) + r_{t+\tau}(0) - 2r_t(\tau) + \left[\sum_{j=t+1}^{t+W} (x_j - x_{j+\tau})\right]^2 \tag{13}
   \]  
4. **Periodic noise**: Joint estimation of \(T\) and interference period \(U\)  

---

## VII. Relations with Auditory Perception  
- **Cancellation model**: Neural equivalent of difference function (Fig. 8)  
- **FFT implementation**: Sum of ACF terms parallels neural processing  
- **Insight**: Auditory models benefit from signal processing optimizations  

---

## VIII. Discussion  
**Key innovations**:  
- Cumulative mean normalization (Step 3) → decouples "too high"/"too low" errors  
- Difference function (Step 2) + threshold (Step 4) → robust subharmonic rejection  
- **Theoretical accuracy**: Not limited by \(\Delta T \Delta F = \text{const}\) (requires ≥27+1 samples)  

**Applications**: Music (wide \(F_0\) range, low latency), polyphony (Sec. VI D)  

---

## IX. Conclusion  
YIN reduces \(F_0\) estimation errors by:  
1. Replacing ACF with difference function  
2. Cumulative mean normalization  
3. Absolute thresholding  
4. Local search refinement  
→ **3× lower errors** vs. best alternatives, **no upper \(F_0\) limit**, minimal tuning.  

---

## Acknowledgments  
Funded by French Ministry of Research (Cognitique) and ATR/CNRS collaboration.  

---

## Appendix: Evaluation Details  
**Databases**:  
1. **DB1**: 28 speakers (Japanese), 0.66h  
2. **DB2**: 2 speakers (English), 0.12h  
3. **DB3**: 4 speakers (French), 0.46h  
4. **DB4**: 4 speakers (English/Japanese), 0.51h  
5. **DB5**: 10 speakers (English), 0.15h  

**Reference Methods**:  
- Praat (`ac`, `cc`, `shs`)  
- Edinburgh PDA (`pda`)  
- Speech Filling System (`fxac`, `fxcep`)  
- Custom (`additive`, `acf`, `nacf`, `TEMPO`)  

**Procedure**:  
- Gross error = \(|F_0_{\text{est}} / F_0_{\text{truth}} - 1| > 0.2\)  
- Time-aligned to laryngograph signal  
- Voicing detection/postprocessing disabled  
