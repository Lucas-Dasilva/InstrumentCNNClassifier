"""
How sound is represented by the computer

1. We use a michrophone
  A microphone has a plastic disc inside it, which is attached to a copper coil and this copper coil is surrounded by a magnet. As the sound waves hit the disc, the coil moves up and down and the movements of the coil past the magnet creates an electrical current.

  The electrical current is the sound signal. Speakers do the opposite. The electrical current creates a signal that moves the plastic disc, creating sound waves.
2. The ADC (Analogue digital converter) takes the analogue wave and maps the data at certain intervals to memory. Specifically 44,100 measurements every single second for cd quality

3. Each data point is 16 bits which maps the potential different 65,536 digits.

"""
"""
Terms & Definitions:

- Sampling frequency or Sampling rate:
  How many times the data is sampled per second which is measured in Hertz(Hz). if sampling frequency is 2 Hz, then two samples are measured in one second.

- Sample size(Bit Depth) or Sampling resolution:
  How many bits are available for each sample. Also known as the bit depth. How many bits can we allocated for each data point or sample. Higher sample size increases range of audio frequency.

- Volume:
  The amplitude of the frequency wave

- Bit rate:
  The number of bits used per second. Bit rate (bps) = Sampling frequency (hz) * Sample size (b)
  Higher bit rate leads to more detailed audio but more data is allocated.

- File size (bits):
  Total size after processing. File size = Sampling frequency (Hz) * Sample size(b) * Time(s)

- Window Lenght or Window size(ms):
  Represents the number of samples, and a duration. Window length = Step size + overlap size 

- Step size (ms):
  How long to move the window size to gather data for entire audio file. The first 400 sample fram starts at sample 0. The next 400 sample frame starts at sample 160 and so on until the end of the audio file is reached.

- Mel Filterbank:
  Filter the audio so that it more accurately represents what humans can hear. Do this by taking the log of the function which in turn focuses more on the lower frequency values and ignores the higher frequencies.
  Standard is 26 filters
  This builds features based on the peaks in the spectrogram
  Forms a 26 x 100 matrix mapping features. so 2600 pixels or features
"""

"""
- We want to apply FFT to sampled data to create a periodagram.
- What is a Periodagram:
  It is an estimate of the spectral density of a signal. So it maps the sample to plot showing the magnitude over frequency. Shows which frequency is most used (usually lower ones) which can be used to down sample our data.
  Tells us which frequencies are present in that frame(kind of like our ear's cochlea)
Example:
Short time fourier transform
  - Sampling rate is 16 kHz (16,000 samples per second collected)
  - Window length is 25 ms 
  - Total samples collected in windown lenght = 400 samples (16,000 * .025)
  - Step Size is 10 ms = 160 samples
"""
"""
- Why used wav files?
  Because it is uncompressed audio. Represents 16 bit audio (-32,768 to 32,767) int

- Signal Envelope:
  Fixes issue where a lot of the signals will be the same. We want to remove dead space that doesn't serve much of a purpose.
  Take absolute value and compute the rolling window with a max value as argument. Finds the maximum amplitude within window and envelopes that window.

- What is Kapre:
  A library created by a spotify employee, to reduce cognitive load. Basically reducing how much code is needed to be written for simple use cases.
  A wrapper for tensorflow.
  Without it we would have to compute the spectrogram offline and use a lot of disk space to store them, and then we would be able to train on it.
  Makes it so that if we need to change parameters then we can simply edit the layer without having to recompute the spectrograms. 
  It basically computes the spectrogram and trains within the sasme scope with only marginal increase to training time

"""