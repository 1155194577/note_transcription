# PYIN: A fundamental frequency estimator using probabilistic threshold distributions

###### Abstract
We propose the Probabilistic YIN (PYIN) algorithm, a modification of the well-known YIN algorithm for fundamental frequency (F0) estimation. Conventional YIN is a simple yet effective method for frame-wise monophonic F0 estimation and remains one of the most popular methods in this domain. In order to eliminate short-term errors, outputs of frequency estimators are usually post-processed resulting in a smoother pitch track. One shortcoming of YIN is that such post-processing cannot fall back on alternative interpretations of the signal because the method outputs precisely one estimate per frame. To address this problem we modify YIN to output multiple pitch candidates with associated probabilities (PYIN Stage 1). These probabilities arise naturally from a prior distribution on the YIN threshold parameter. We use these probabilities as observations in a hidden Markov model, which is Viterbi-decoded to produce an improved pitch track (PYIN Stage 2). We demonstrate that the combination of Stages 1 and 2 raises recall and precision substantially. The additional computational complexity of PYIN over YIN is low. We make the method freely available online as an open source C++ library for Vamp hosts.

**Keywords**: Pitch estimation, pitch tracking, YIN

**Authors**: Matthias Mauch and Simon Dixon  
Queen Mary University of London, Centre for Digital Music, Mile End Road, London  

**Source**: [https://code.soundsoftware.ac.uk/projects/pyin](https://code.soundsoftware.ac.uk/projects/pyin)

---

## 1 Introduction
The estimation of the fundamental frequency (F0) from monophonic human voice signals is a prerequisite to comprehensive analysis of intonation in speech [1] and singing [2]. Since frame-wise pitch estimates are not completely reliable, post-processing is often used to clean the raw pitch track (see, e.g. [3]). Despite the high success rate of existing algorithms, this procedure is generally flawed because (potentially correct) frequency candidates that were discarded in the frame-wise stage cannot be recovered in the smoothing stage, and hence multiple frame-wise pitch candidates should be used before smoothing [4].

Several solutions to the problem of F0 estimation have been proposed in the area of speech processing [4, 5, 6, 7]. Among these, the YIN fundamental frequency estimator [7] has gained popularity beyond the speech processing community, especially in the analysis of singing [8, 9]. Babacan et al. [10] provide an overview of the performance of F0 trackers on singing, in which YIN is shown to be state of the art, and particularly good at fine pitch recognition.

The original YIN paper outlines a smoothing procedure that does not use the frame-wise estimate, but tracks low values in the underlying periodicity function. In this paper, we take YIN and modify its frame-wise variant in a probabilistic way to output multiple pitch candidates with associated probabilities, hence also reducing the loss of useful information before smoothing. We then employ a hidden Markov model which uses the modified frame-wise output to calculate a smoothed pitch track which retains YIN's well-known pitch accuracy and at the same time can be shown to provide excellent recall and precision.

---

## 2 Method
This section describes PYIN, our proposed method, which is divided into two stages:  
(1) frame-wise extraction of multiple pitch candidates with associated probabilities, and  
(2) HMM-based tracking of the pitch candidates into a monophonic pitch track.

### 2.1 Stage 1: F0 Candidates
The first stage of PYIN follows the same steps as the original YIN algorithm, differing only in the thresholding stage, where it assumes a threshold distribution, in contrast to YIN, which relies on a single threshold (see Fig. 1).

The YIN algorithm is based on the intuition that, in a signal \(x_{i}, i=1,\ldots,2W\), the difference  

$$d_{t}(\tau)=\sum_{j=1}^{W}(x_{j}-x_{j+\tau})^{2},$$  

will be small if the signal is approximately periodic with fundamental period \(\tau=1/f_{0}\). It can be shown [7] that this difference can be conveniently obtained by first calculating the auto-correlation function (ACF)  

$$r_{t}(\tau)=\sum_{j=t+1}^{t+W}x_{j}x_{j+\tau},$$  

from which (1) can be calculated as  

$$d_{t}(\tau)=r_{t}(0)+r_{t+\tau}(0)-2r_{t}(\tau).$$  

These two calculations comprise the first two steps of the original YIN algorithm. The third step is a normalisation of the difference (1) obtaining a 'cumulative mean normalised difference function' \(d^{\prime}(\tau)\) (subscript \(t\) omitted for simplicity).

The fourth step of YIN is to find the dip in \(d^{\prime}\) corresponding to the fundamental period by picking the smallest \(\tau\) where \(d^{\prime}\) has a local minimum and \(d^{\prime}(\tau)<s\) for a fixed threshold \(s\) (usually \(s=0.1\) or \(s=0.15\)). We notate the period estimated by YIN as \(Y(x_{t},s)\).

Since both the choice of \(s\) and the strategy for handling cases where all values exceed the threshold affect the result, we propose using a prior parameter distribution \(S\) given by \(P(s_{i})\), where \(s_{i}, i=1,\ldots,N\) are possible thresholds (thresholds: 0.01 to 1.0 in steps of 0.01, \(N=100\)). The distributions used are Beta distributions with means 0.1, 0.15, 0.2 (\(\alpha=1\) and \(\beta=18,11\frac{1}{3},8\)). Given this distribution and prior probability \(p_{a}\) of using the absolute minimum strategy, the probability that period \(\tau\) is fundamental period \(\tau_{0}\) is:

$$P(\tau=\tau_{0}|S,x_{t})=\sum_{i=1}^{N}a(s_{i},\tau)\,P(s_{i})\,[Y(x_{t},s_{i})=\tau],$$

where \([\cdot]\) is the Iverson bracket (1 if true, 0 otherwise), and  

$$a(s_{i},\tau)=\begin{cases}1, &\text{if }d^{\prime}(\tau)<s_{i}\\ p_{a}, &\text{otherwise.}\end{cases}$$  

(We use \(p_{a}=0.01\).) Any \(\tau\) with \(P(\tau=\tau_{0}|S,x_{t})>0\) yields a fundamental frequency candidate \(f=1/\tau\). Frequency estimates are improved by parabolic interpolation on \(d^{\prime}\).

**Properties of Stage 1 Output**:
1. Every candidate \(\tau\) is a genuine YIN estimate for some threshold \(s\)
2. Minimal computational overhead vs. original YIN
3. Conventional YIN estimate is included in candidates

### 2.2 Stage 2: HMM-based Pitch Tracking
The pitch tracking step chooses at most one pitch candidate per frame. We divide pitch space into \(M=480\) bins covering four octaves (55Hz–880Hz) in 10-cent steps. These bins are modelled as states in a hidden Markov model (HMM). The model uses probabilities from Stage 1 as observation probabilities, resulting in a sparse observation vector \(p_{m}^{*},m=1,\ldots,M\).

We extend the HMM with voiced (\(v=1\)) and unvoiced (\(v=0\)) states per pitch (\(2M\) total states). Assuming \(P(v=1)=P(v=0)=0.5\), observation probability is:  

$$p_{m,v}=\begin{cases}0.5\cdot p_{m}^{*}, &\text{for }v=1\\ 0.5\cdot(1-\sum_{k}P_{k}^{*}) &\text{for }v=0.\end{cases}$$  

Transition probabilities enforce smooth pitch tracks and few voicing changes:  

$$\begin{split}
p_{v}&=P(v_{t}\mid v_{t-1})\\
&=\begin{cases}0.99, &\text{if no change}\\ 0.01 &\text{otherwise}
\end{cases}\\
p_{ij}&=P(\text{pitch}_{t}=j\mid\text{pitch}_{t-1}=i)
\end{split}$$  

Pitch transitions use a triangular distribution allowing jumps ≤25 bins (2.5 semitones/frame). The HMM is decoded using an efficient Viterbi algorithm.

---

## 3 Results
We evaluated PYIN on >30 hours of synthesized singing (RWC Music Database [12]) degraded via Audio Degradation Toolbox [13]. Tested configurations:
- **PYIN**: Three versions with Beta distributions (means 0.10, 0.15, 0.20)
- **YIN+S**: Original YIN + smoothing (\(s=0.10, 0.15, 0.20\))
- **Baseline**: Conventional YIN (\(s=0.10, 0.15, 0.20\))

### 3.1 Quantitative Analysis
**Recall** (proportion of correctly tracked voiced frames, ±1 semitone tolerance):  
PYIN substantially outperforms YIN (medians: PYIN .10=0.977, PYIN .15=0.982, PYIN .20=0.984 vs YIN≤0.951).  

**Precision & F Score**:  
PYIN and YIN+S outperform baseline YIN in precision. PYIN achieves optimal F score (balance of precision/recall).  

**Robustness**:  
PYIN shows excellent resistance to octave errors (0.5%–1.7%) and strong voicing detection (recall: 92.5%–95.0%, specificity: 88.9%–91.9%).

**Table 1**: Recall by YIN parameter and degradation
|          | orig. | noise | sound | live rec. | phone rec. | clip. |
|----------|-------|-------|-------|-----------|------------|-------|
| full cand. | **0.993** | **0.993** | **0.990** | **0.750** | **0.953** | **0.985** |
| YIN .10  | 0.988 | 0.980 | 0.886 | **0.648** | **0.880** | **0.979** |
| YIN .15  | **0.989** | **0.988** | 0.934 | 0.644 | 0.843 | 0.958 |
| YIN .20  | 0.980 | 0.986 | **0.953** | 0.632 | 0.803 | 0.931 |

### 3.2 Qualitative Example
Figure 4 compares pitch tracks for "Happy Birthday" (female singer). YIN exhibits octave errors (likely due to breathy timbre), while PYIN produces a correct, robust track.

---

## 4 Conclusion
We presented PYIN, a probabilistic modification of YIN that:  
1. Generates multiple pitch candidates per frame via threshold distribution  
2. Uses HMM smoothing for temporal consistency  
PYIN significantly improves recall/precision over YIN while adding minimal computational overhead. Its distribution-based parametrization provides greater robustness than YIN's single-threshold approach.

---

## References
[1] Fang Liu and Yi Xu, "Question Intonation as Affected by Word Stress and Focus in English," in _Proceedings of the 16th International Congress of Phonetic Sciences_, 2007.  
[2] Johanna Devaney and Daniel P W Ellis, "An Empirical Approach to Studying Intonation Tendencies in Polyphonic Vocal Performances," _Journal of Interdisciplinary Music Studies_, 2008.  
[3] Baris Bozkurt, "An Automatic Pitch Analysis Method for Turkish Maqam Music," _Journal of New Music Research_, 2008.  
[4] David Talkin, "A Robust Algorithm for Pitch Tracking," in _Speech Coding and Synthesis_, 1995.  
[5] Paul Boersma, "Praat, a system for doing phonetics by computer," _Glot International_, 2001.  
[6] Hideki Kawahara et al., "Aperiodicity extraction and control... system STRAIGHT," _Proceedings of MAVEBA_, 2001.  
[7] Alain de Cheveigne and Hideki Kawahara, "YIN, a fundamental frequency estimator for speech and music," _The Journal of the Acoustical Society of America_, 2002.  
[8] Johanna Devaney and Dan Ellis, "Improving MIDI-Audio Alignment with Audio Features," in _IEEE Workshop on Applications of Signal Processing to Audio and Acoustics_, 2009.  
[9] Matti P Ryynanen, _Probabilistic Modelling of Note Events..._, Ph.D. thesis, 2004.  
[10] Onur Babacan et al., "A Comparative Study of Pitch Extraction Algorithms...," in _ICASSP_, 2013.  
[11] Joren Six and Olmo Cornelis, "Tarsos - A Platform to Explore Pitch Scales...," in _ISMIR_, 2011.  
[12] Masataka Goto et al., "RWC Music Database...," in _ISMIR_, 2002.  
[13] Matthias Mauch and Sebastian Ewert, "The Audio Degradation Toolbox...," in _ISMIR_, 2013.  
[14] Matthias Mauch et al., "Intonation in Unaccompanied Singing...," _under review_, 2013.

---

**Figures**  
*Fig. 1*: Comparison of YIN vs PYIN workflow  
*Fig. 2*: Beta distributions used as parameter priors  
*Fig. 3*: Box plots of performance measures (recall/precision/F score)  
*Fig. 4*: Pitch tracks of PYIN vs YIN on "Happy Birthday"