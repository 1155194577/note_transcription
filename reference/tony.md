# COMPUTER-AIDED MELODY NOTE TRANSCRIPTION USING THE TONY SOFTWARE: ACCURACY AND EFFICIENCY

**Matthias Mauch**  
Queen Mary University of London  
**George Fazekas**  
Queen Mary University of London  
**Juan Bello**  
New York University  
**Chris Cannam**  
Queen Mary University of London  
**Rachel Bittner**  
New York University  
**Justin Salamon**  
New York University  
**Jailie Dai**  
Queen Mary University of London  
**Simon Dixon**  
Queen Mary University of London  

---

### Abstract  
We present Tony, a software tool for interactive annotation of melodies from monophonic audio recordings, evaluating its usability and note extraction accuracy. Scientific study of acoustic performances requires accurate note/pitch transcription. Researchers typically manually correct automatic transcription results, and Tony streamlines this process by providing:  
(a) State-of-the-art pitch/note estimation algorithms  
(b) Visual/auditory feedback for error-spotting  
(c) Intelligent GUI for rapid error correction  
(d) Extensive export functions  

Tony's automatic transcription outperforms existing tools. Annotation time was measured on 96 solo vocal recordings, considering piece complexity, edit frequency, and user expertise. Tony is open-source (Windows/Mac/Linux) at [https://code.soundsoftware.ac.uk/projects/tony/](https://code.soundsoftware.ac.uk/projects/tony/).

---

## 1 Introduction  
Goal: Improve efficiency in melodic annotation, especially for singing. While existing DSP methods measure pitch accurately (e.g., [1,2]), manual adjustments remain necessary for discrete note transcription. This paper contributes:  
1. **Tony UI** for streamlined melody annotation  
2. **New note transcription algorithm** (pYIN Vamp plugin)  
3. **Evaluation** of accuracy and annotation effort  

Features reflect Tony v1.0 unless specified.

---

## 2 Background  
Researchers use software to annotate:  
- **Pitch tracks**: Continuous F0 contours  
- **Note tracks**: Discrete note sequences  

**Survey Findings** (31 respondents):  
- 87% academic, 35% students, 13% industry  
- 58% experience with pitch tracks, 52% with note tracks  

**Popular Tools**:  
- UI-based: Sonic Visualiser [3], Praat [4]  
- DSP-based: YIN [5]  
Limitations: Laborious note extraction, toolchain complexity (e.g., AMPACT alignment [7]), proprietary solutions (Melodyne) lack transparency/export.  

**Conclusion**: Need for dedicated melodic annotation tool. Tony provides:  
(a) High-resolution pitch/note algorithms  
(b) Visual/auditory feedback  
(c) Interactive error correction  
(d) Export capabilities  
(e) Open-source availability  

**Pitch Estimation**:  
- Focus: Monophonic audio (not polyphonic melody extraction [9,10])  
- State-of-the-art: YIN [5] for singing [14,15], pYIN robust against degradations [17]  

**Note Transcription**:  
- Less studied than pitch tracking  
- Tony’s novel method (first evaluation in this paper)  

**Software Availability**:  
| Software         | URL                                      |
|------------------|------------------------------------------|
| Tony             | https://code.soundsoftware.ac.uk/projects/tony |
| pYIN             | https://code.soundsoftware.ac.uk/projects/pyin |
| Sonic Visualiser | https://code.soundsoftware.ac.uk/projects/sv  |

---

## 3 Method  
Tony processes audio at 44.1 kHz using 2048-sample frames (≈46 ms) with 256-sample hop (≈6 ms).

### 3.1 Pitch Estimation  
Uses **pYIN** [17]:  
1. Stage 1: Multiple pitch candidates with probabilities  
2. Stage 2: Viterbi-decoded HMM for refined pitch track  

### 3.2 Note Transcription  
Input: pYIN pitch track  
Output: Discrete notes via Viterbi-decoded HMM  
- **Pitch range**: MIDI 35 (B1 ≈61 Hz) to 85 (C6 ≈1109 Hz) at 3 steps/semitone (207 pitches)  
- **States per pitch**: Attack, Stable, Silence  

**Emission likelihood** for pitch \(q\) given state \(n_p\) (MIDI pitch \(p\)):  
\[
P(n_p | q) = v \cdot \left( \frac{1}{z} \left[ \phi_{p,\sigma}(q) \right]^\tau \right)
\]  
Where:  
- \(\phi_{p,\sigma}\): Gaussian PDF centered at \(p\)  
- \(\sigma = 5\) semitones (Attack), \(\sigma = 0.9\) (Stable)  
- \(\tau = 0.1\) (trust in pitch estimate)  
- \(v = 0.5\) (prior voiced probability)  

**Transition model**:  
- Left-to-right HMM within notes (Attack → Stable → Silence)  
- Pitch transitions constrained:  
  - Max 13-semitone interval  
  - Small changes favored  
  - ≥⅔ semitone difference required  

### 3.3 Note Post-processing  
1. **Amplitude-based onset segmentation**:  
   - Detect amplitude rise \( r = \frac{a_{i+1}}{a_{i-1}} \)  
   - If \( 1/r < s \) (sensitivity), mark frame as unvoiced  
2. **Minimum duration pruning**: Discard notes <100 ms  

### 3.4 Semi-automatic Pitch Re-estimation  
Users select:  
1. **Multiple pYIN tracks** (13 frequency ranges):  
   \[
   p_{ij} = p_i \times \phi_{c_j,\sigma_r}(m_i) \quad (c_j = 48 + 3j;\, \sigma_r = 8)
   \]  
   Eliminate duplicates (>80% pitch overlap).  
2. **Time-pitch rectangle**: Harmonic product spectrum [23] within user-defined bounds.  

---

## 4 User Interface  
**Key features**:  
- Integrated visualization: Pitch/note tracks, spectrogram, waveform  
- Non-overlapping note design prevents errors  
- Real-time pitch calculation (median of underlying pitch track)  

**Editing**:  
- **Pitch actions**: Octave shift, removal, alternative tracks  
- **Note actions**: Split, merge, delete, create, move boundaries  

**Auditory feedback**:  
- **Notes**: Wavetable electric piano (neutral timbre)  
- **Pitch track**: Sinusoidal synthesis (first 3 harmonics)  

---

## 5 Evaluation  
### 5.1 Automatic Transcription Accuracy  
**Dataset**: 38 solo vocal pieces [24]  

**Comparison methods**:  
- Melotranscript [22]  
- Gomez & Bonada [2]  
- Ryynanen [21]  

**Metrics**:  
- Frame-wise: Overall accuracy, raw pitch accuracy, voicing recall  
- Note-based: COnPOff (\(F\)), COnP (\(F\)), COn (\(F\))  

**Results (Table 3)**:  
- pYIN achieved **best overall accuracy (0.85)**  
- Optimal parameters: Pruning ≥100 ms, sensitivity \(s ≥ 0.6\)  
- Post-processed pYIN improved COnPOff \(F\) to **0.50** (vs. 0.45 for Melotranscript)  

### 5.2 Manual Correction Effort  
**Dataset**: 96 amateur vocal recordings  

**Findings**:  
- Dominant edits: Join (8.64/rec), Delete (8.82/rec)  
- Time-consuming edits: Create (145 sec/edit), Split (5.58 sec/edit)  
- Familiarity reduced annotation time (-2.31 sec/day)  

**Regression model** (\(R^2 = 25\%\)):  
| Factor      | Time (sec) | \(p\)-value |
|-------------|------------|------------|
| Intercept   | 437.20     | <0.01      |
| Create      | 145.08     | <0.01      |
| Delete      | 3.51       | 0.06       |
| Split       | 5.58       | 0.06       |
| Join        | 3.18       | 0.18       |
| Move        | 45.51      | 0.25       |
| Familiarity | -2.31      | 0.01       |

---

## 6 Discussion  
- **Edit cost**: Splits > Merges → Prioritize onset recall in auto-transcription  
- **Accuracy dependencies**: Instrumentation, voicing prevalence  
- **Adoption**: Used in MedleyDB [27] and unpublished studies  

---

## 7 Conclusions  
- pYIN note transcription is **state-of-the-art**  
- Tony enables **efficient annotation** (open-source)  
- Future work: Optimize for costly operations (splits/creates)  

---

### Acknowledgements  
Supported by Royal Academy of Engineering (Mauch).  

---

### Appendix A: Survey Questions  
1. Have you used pitch annotation software? (multi-choice)  
2. List known pitch annotation tools. (free text)  
3. Desired tool improvements? (free text)  
4. Comments? (free text)  
5. Field of work? (multi-choice)  

---

## References  
[1] Pant et al., 2010; [2] Gomez & Bonada, 2013; [3] Cannam et al., 2006; [4] Boersma, 2001; [5] de Cheveigné & Kawahara, 2002; ... [27] Bittner et al., 2014 *(Full list in original PDF)*