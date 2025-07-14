mod traits;

use atomic_float::AtomicF32;
use hound::{SampleFormat, WavSpec};
use rodio::{ChannelCount, OutputStreamBuilder, Source};
use std::path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use petgraph::algo::toposort;
use petgraph::graph::{Graph, NodeIndex};
use traits::{AudioNode, Sample};

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
    modulator_input: Option<f32>,
    active: bool,
}

impl WavetableOscillator {
    fn new(
        sample_rate: u32,
        wave_table: Vec<f32>,
        amplitude: Arc<AtomicF32>,
        frequency: Arc<AtomicF32>,
    ) -> WavetableOscillator {
        WavetableOscillator {
            sample_rate,
            wave_table,
            index: 0.0,
            amplitude,
            frequency,
            modulator_input: None,
            active: true,
        }
    }

    fn set_modulator_input(&mut self, input: f32) {
        self.modulator_input = Some(input);
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

impl AudioNode for WavetableOscillator {
    fn compute(&mut self) -> Sample {
        if !self.active {
            return 0.0;
        }

        let frequency = self.frequency.load(Ordering::Relaxed);
        let amplitude = self.amplitude.load(Ordering::Relaxed);

        // Get modulator input if available
        let modulator_sample = self.modulator_input.unwrap_or(0.0);

        // Calculate final phase as carrier phase + modulator signal
        let increment = frequency + modulator_sample;
        let index_increment = increment * self.wave_table.len() as f32 / self.sample_rate as f32;

        let sample = self.lerp();
        self.index += index_increment;
        self.index %= self.wave_table.len() as f32;

        amplitude * sample
    }

    fn reset(&mut self) {
        self.index = 0.0;
        self.modulator_input = None;
    }
}

// Iterator implementation required to be a rodio Source
impl Iterator for WavetableOscillator {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        Some(self.compute())
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
        None
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

pub fn output_to_wav(
    source: &mut impl Source,
    wav_file: impl AsRef<path::Path>,
    duration: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let format = WavSpec {
        channels: source.channels() as ChannelCount,
        sample_rate: source.sample_rate(),
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(wav_file, format)?;

    let total_samples = (duration.as_secs_f32() * source.sample_rate() as f32) as usize;

    for _ in 0..total_samples {
        if let Some(sample) = source.next() {
            writer.write_sample(sample)?;
        } else {
            break;
        }
    }

    writer.finalize()?;
    Ok(())
}

#[allow(non_snake_case)]
fn main() {
    let wave_table_size = 64;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);

    // Render sine to wavetable
    for n in 0..wave_table_size {
        wave_table.push((2.0 * std::f32::consts::PI * n as f32 / wave_table_size as f32).sin());
    }

    // Define parameters
    let A_C = Arc::new(AtomicF32::new(1.0));
    let I = 16.0;
    let R_f = 14.0;
    let f_C = Arc::new(AtomicF32::new(440.0));
    let f_M = Arc::new(AtomicF32::new(f_C.load(Ordering::Relaxed) / R_f));
    let A_M = Arc::new(AtomicF32::new(I * f_M.load(Ordering::Relaxed)));

    let modulator = WavetableOscillator::new(
        44100,
        wave_table.clone(),
        Arc::clone(&A_M),
        Arc::clone(&f_M),
    );
    let carrier = WavetableOscillator::new(44100, wave_table, Arc::clone(&A_C), Arc::clone(&f_C));

    // Create the audio graph
    let mut graph = Graph::<WavetableOscillator, ()>::new();

    // Add nodes to the graph
    let modulator_node = graph.add_node(modulator);
    let carrier_node = graph.add_node(carrier);

    // Connect modulator output to carrier input
    graph.add_edge(modulator_node, carrier_node, ());

    // Create a graph-based source for rodio
    struct GraphSource {
        graph: Graph<WavetableOscillator, ()>,
        sorted_nodes: Vec<NodeIndex>,
        connections: Vec<(NodeIndex, NodeIndex)>,
        carrier_node: NodeIndex,
        sample_rate: u32,
    }

    impl Iterator for GraphSource {
        type Item = f32;

        fn next(&mut self) -> Option<f32> {
            // Process each node in pre-computed order
            for node_index in &self.sorted_nodes {
                // Transfer data to connected nodes
                for (from_node, to_node) in &self.connections {
                    if from_node == node_index {
                        let modulator_sample = self.graph[*from_node].compute();
                        self.graph[*to_node].set_modulator_input(modulator_sample);
                    }
                }
            }

            // Return the final output (from carrier)
            let carrier = &mut self.graph[self.carrier_node];
            Some(carrier.compute())
        }
    }

    impl Source for GraphSource {
        fn channels(&self) -> u16 {
            1
        }

        fn sample_rate(&self) -> u32 {
            self.sample_rate
        }

        fn current_span_len(&self) -> Option<usize> {
            None
        }

        fn total_duration(&self) -> Option<Duration> {
            None
        }
    }

    // Compute topological sort once
    let sorted_nodes = toposort(&graph, None).expect("Graph has cycles");

    // Create graph source
    let graph_source = GraphSource {
        graph,
        sorted_nodes: sorted_nodes.clone(),
        connections: vec![(modulator_node, carrier_node)],
        carrier_node,
        sample_rate: 44100,
    };

    // Render to wav using graph
    output_to_wav(
        &mut GraphSource {
            graph: graph_source.graph.clone(),
            sorted_nodes,
            connections: vec![(modulator_node, carrier_node)],
            carrier_node,
            sample_rate: 44100,
        },
        "output.wav",
        Duration::from_secs(5),
    )
    .expect("Failed to write to wav");

    // play audio
    let Ok(stream_handle) = OutputStreamBuilder::open_default_stream() else {
        todo!()
    };

    stream_handle.mixer().add(graph_source);

    // adjust parameters over time
    for _i in 0..1000 {
        let mut current_freq = f_C.load(Ordering::Relaxed);
        current_freq += 1.0;
        f_C.store(current_freq, Ordering::Relaxed);
        f_M.store(current_freq / R_f, Ordering::Relaxed);
        A_M.store(I * f_M.load(Ordering::Relaxed), Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(5));
    }
}
