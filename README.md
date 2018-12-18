# LeadsheetVAE 自動簡譜變奏 :musical_note:
[Lead Sheet VAE](https://liuhaumin.github.io/LeadsheetArrangement/) is a task to automatically generate and variate lead sheets. There are several types we use in generation and variation.
- **Unconditional generation:** generate melody and chords from nothing (sampling from Gaussian noise)
- **Conditional generation:**
  - generate lead sheets conditioned on chords
  - generate lead sheets conditioned on melody contour
  - generate lead sheets conditioned on melody contour and chords
- **variations:**
  - random variations
  - melody similar variations
  - chord similar variations
  - song mashup using interpolation
  - variations conditioned on style/ emotion

We train the model with TheoryTab (TT) dataset to generate pop song style leadsheets.

Sample results are available
[here](https://liuhaumin.github.io/LeadsheetArrangement/results).

## Papers

__Lead Sheet Generation and Arrangement via a Hybrid Generative Model__<br>
Hao-Min Liu\*, Meng-Hsuan Wu\*, and Yi-Hsuan Yang
(\*equal contribution)<br>
in _ISMIR Late-Breaking Demos Session_, 2018.
(non-refereed two-page extended abstract)<br>
[[paper](https://liuhaumin.github.io/LeadsheetArrangement/pdf/ismir2018leadsheetarrangement.pdf)]
[[poster](https://liuhaumin.github.io/LeadsheetArrangement/pdf/ismir-lbd-poster_A0_final.pdf)]
