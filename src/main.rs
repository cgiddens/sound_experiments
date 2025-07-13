use atomic_float::AtomicF32;
use rodio::{OutputStreamBuilder, Source};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

// This will eventually be an FM oscillator...

// Modulation Index I:
// I = A_M / f_M

// Frequency Ratio R_f
// R_f = f_C / f_M

// FM (PM) output:
// s_PM(t) = A_C * cos(2 * pi * f_C * t + I * sin(2 * pi * f_M * t))

// to maintain timbre, I and R_f must be constant
// so,
// s_PM(t) = A_C * cos(2 * pi * f_C * t + I * sin(2 * pi * (f_C / R_f) * t))

struct WavetableOscillator {
    sample_rate: u32,
    wave_table: Vec<f32>,
    index: f32,
    frequency: Arc<AtomicF32>,
}

impl WavetableOscillator {
    fn new(
        sample_rate: u32,
        wave_table: Vec<f32>,
        frequency: Arc<AtomicF32>,
    ) -> WavetableOscillator {
        WavetableOscillator {
            sample_rate: sample_rate,
            wave_table: wave_table,
            index: 0.0,
            frequency,
        }
    }

    fn get_sample(&mut self) -> f32 {
        let frequency = self.frequency.load(Ordering::Relaxed);
        let index_increment = frequency * self.wave_table.len() as f32 / self.sample_rate as f32;

        let sample = self.lerp();
        self.index += index_increment;
        self.index %= self.wave_table.len() as f32;
        return sample;
    }

    fn lerp(&self) -> f32 {
        let truncated_index = self.index as usize;
        let next_index = (truncated_index + 1) % self.wave_table.len();

        let next_index_weight = self.index - truncated_index as f32;
        let truncated_index_weight = 1.0 - next_index_weight;

        truncated_index_weight * self.wave_table[truncated_index]
            + next_index_weight * self.wave_table[next_index]
    }
}

impl Iterator for WavetableOscillator {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        Some(self.get_sample())
    }
}

impl Source for WavetableOscillator {
    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn current_span_len(&self) -> Option<usize> {
        None // never stops playing
    }

    fn total_duration(&self) -> Option<Duration> {
        None // never stops playing
    }
}

fn main() {
    let wave_table_size = 64;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);

    for n in 0..wave_table_size {
        wave_table.push((2.0 * std::f32::consts::PI * n as f32 / wave_table_size as f32).sin());
    }

    let frequency = Arc::new(AtomicF32::new(440.0));
    let oscillator = WavetableOscillator::new(44100, wave_table, Arc::clone(&frequency));

    let Ok(stream_handle) = OutputStreamBuilder::open_default_stream() else {
        todo!()
    };

    stream_handle.mixer().add(oscillator);

    for _i in 0..1000 {
        let current_freq = frequency.load(Ordering::Relaxed);
        frequency.store(current_freq + 1.0, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(5));
    }
}
