use atomic_float::AtomicF32;
use rodio::{OutputStreamBuilder, Source};
use std::sync::Arc;
use std::sync::Mutex;
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

#[derive(Clone)]
struct WavetableOscillator {
    sample_rate: u32,
    wave_table: Vec<f32>,
    index: f32,
    amplitude: Arc<AtomicF32>,
    frequency: Arc<AtomicF32>,
    modulator: Option<Arc<Mutex<WavetableOscillator>>>,
}

impl WavetableOscillator {
    fn new(
        sample_rate: u32,
        wave_table: Vec<f32>,
        amplitude: Arc<AtomicF32>,
        frequency: Arc<AtomicF32>,
        modulator: Option<Arc<Mutex<WavetableOscillator>>>,
    ) -> WavetableOscillator {
        WavetableOscillator {
            sample_rate: sample_rate,
            wave_table: wave_table,
            index: 0.0,
            amplitude,
            frequency,
            modulator,
        }
    }

    fn get_sample(&mut self) -> f32 {
        let frequency = self.frequency.load(Ordering::Relaxed);
        let amplitude = self.amplitude.load(Ordering::Relaxed);

        let modulator_sample = if let Some(modulator) = &self.modulator {
            modulator.lock().unwrap().get_sample()
        } else {
            0.0
        };

        let increment = frequency + modulator_sample;
        let index_increment = increment * self.wave_table.len() as f32 / self.sample_rate as f32;

        let sample = self.lerp();
        self.index += index_increment;
        self.index %= self.wave_table.len() as f32;
        return amplitude * sample;
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

    let A_C = Arc::new(AtomicF32::new(1.0));
    let I = 1.0; // A_M = f_M in this case only
    let R_f = 7.0; // modulator 7x below carrier
    let f_C = Arc::new(AtomicF32::new(440.0));
    let f_M = Arc::new(AtomicF32::new(f_C.load(Ordering::Relaxed) / R_f));
    let A_M = Arc::new(AtomicF32::new(I * f_M.load(Ordering::Relaxed)));

    let modulator = WavetableOscillator::new(
        44100,
        wave_table.clone(),
        Arc::clone(&A_M),
        Arc::clone(&f_M),
        None,
    );
    let carrier = WavetableOscillator::new(
        44100,
        wave_table,
        Arc::clone(&A_C),
        Arc::clone(&f_C),
        Some(Arc::new(Mutex::new(modulator))),
    );

    let Ok(stream_handle) = OutputStreamBuilder::open_default_stream() else {
        todo!()
    };

    stream_handle.mixer().add(carrier);

    for _i in 0..1000 {
        let mut current_freq = f_C.load(Ordering::Relaxed);
        current_freq += 1.0;
        f_C.store(current_freq, Ordering::Relaxed);
        f_M.store(current_freq / R_f, Ordering::Relaxed);
        A_M.store(I * f_M.load(Ordering::Relaxed), Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(5));
    }
}
