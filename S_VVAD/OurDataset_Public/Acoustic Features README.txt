Extracted features are:

1) Mel Frequency Cepstral Coefficients

2) Filterbank Energies

3) Log Filterbank Energies

4) Spectral Subband Centroids

using the code, which can be found in  https://github.com/jameslyons/python_speech_features/blob/master/README.rst

..................................................................
1) MFCC Features
..................................................................
- the length of the analysis window in seconds: 25 milliseconds
- sample rate: 8000
- the step between successive windows in seconds: 10 milliseconds
- the number of cepstrum to return: 13
- the number of filters in the filterbank: 26.
- the FFT size: 512
- lowest band edge of mel filters: 0 Hz
- highest band edge of mel filters: samplerate/2 Hz
- preemphasis filter applied is 0.97
- a lifter to final cepstral coefficients: 22

..................................................................
2) Filterbank Features: These filters are raw filterbank energies. 
For most applications you will want the logarithm of these features. 
..................................................................
- sample rate: 8000
- the length of the analysis window in seconds: 25 milliseconds
- the step between seccessive windows in seconds: 10 milliseconds
- the number of filters in the filterbank: 26
- the FFT size: 512
- lowest band edge of mel filters: 0 Hz
- highest band edge of mel filters: samplerate/2 Hz
- preemphasis filter with coefficient: 0.97

