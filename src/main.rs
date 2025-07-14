mod traits;

use atomic_float::AtomicF32;
use hound::{SampleFormat, WavSpec};
use rodio::{ChannelCount, OutputStreamBuilder, Source};
use std::path;
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

use petgraph::algo::toposort;
use petgraph::graph::{Graph, NodeIndex};
use traits::{
    AudioNode, ConcreteInputPort, ConcreteOutputPort, InputPort, OutputPort, Port, Sample,
};

struct WavetableOscillator {
    sample_rate: u32,
    wave_table: Vec<f32>,
    index: f32,
    amplitude: Arc<AtomicF32>,
    frequency: Arc<AtomicF32>,
    inputs: Vec<ConcreteInputPort>,
    outputs: Vec<ConcreteOutputPort>,
    modulator: Option<Arc<Mutex<WavetableOscillator>>>,
    active: bool,
}

impl WavetableOscillator {
    fn new(
        sample_rate: u32,
        wave_table: Vec<f32>,
        amplitude: Arc<AtomicF32>,
        frequency: Arc<AtomicF32>,
        modulator: Option<Arc<Mutex<WavetableOscillator>>>,
    ) -> WavetableOscillator {
        let mut inputs = Vec::new();
        if modulator.is_some() {
            inputs.push(ConcreteInputPort::new(
                "modulator".to_string(),
                sample_rate,
                1,
            ));
        }

        let mut outputs = Vec::new();
        outputs.push(ConcreteOutputPort::new(
            "output".to_string(),
            sample_rate,
            1,
        ));

        WavetableOscillator {
            sample_rate,
            wave_table,
            index: 0.0,
            amplitude,
            frequency,
            inputs,
            outputs,
            modulator,
            active: true,
        }
    }

    fn get_sample(&mut self) -> f32 {
        let frequency = self.frequency.load(Ordering::Relaxed);
        let amplitude = self.amplitude.load(Ordering::Relaxed);

        // Get modulator input from port if available
        let modulator_sample = if let Some(input) = self.inputs.get(0) {
            input.current_data()[0]
        } else {
            0.0
        };

        // calculate final phase as carrier phase + modulator signal
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

// Iterator implementation required to be a rodio Source
impl Iterator for WavetableOscillator {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        Some(self.get_sample())
    }
}

impl AudioNode for WavetableOscillator {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    fn output_count(&self) -> usize {
        self.outputs.len()
    }

    fn input(&self, index: usize) -> Option<&dyn InputPort> {
        self.inputs.get(index).map(|input| input as &dyn InputPort)
    }

    fn input_mut(&mut self, index: usize) -> Option<&mut dyn InputPort> {
        self.inputs
            .get_mut(index)
            .map(|input| input as &mut dyn InputPort)
    }

    fn output(&self, index: usize) -> Option<&dyn OutputPort> {
        self.outputs
            .get(index)
            .map(|output| output as &dyn OutputPort)
    }

    fn output_mut(&mut self, index: usize) -> Option<&mut dyn OutputPort> {
        self.outputs
            .get_mut(index)
            .map(|output| output as &mut dyn OutputPort)
    }

    fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active {
            return Ok(());
        }

        // Compute the next sample
        let sample = self.get_sample();

        // Send to output
        if let Some(output) = self.outputs.get_mut(0) {
            output.send(&[sample]);
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.index = 0.0;
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

impl Source for WavetableOscillator {
    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    // returns # of samples until end of source
    fn current_span_len(&self) -> Option<usize> {
        None // never stops playing
    }

    fn total_duration(&self) -> Option<Duration> {
        None // never stops playing
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

    // Calculate total samples needed
    let total_samples = (duration.as_secs_f32() * source.sample_rate() as f32) as usize;

    // Iterate for exactly the number of samples we need
    for _ in 0..total_samples {
        if let Some(sample) = source.next() {
            writer.write_sample(sample)?;
        } else {
            break; // Source ended early
        }
    }

    writer.finalize()?;
    Ok(())
}

#[allow(non_snake_case)]
fn main() {
    let wave_table_size = 64;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);

    // render sine to wavetable
    for n in 0..wave_table_size {
        wave_table.push((2.0 * std::f32::consts::PI * n as f32 / wave_table_size as f32).sin());
    }

    // define parameters

    // carrier amplitude
    let A_C = Arc::new(AtomicF32::new(1.0));

    // modulation index (A_M / f_M)
    let I = 16.0; // A_M = f_M in this case only

    // frequency ration (f_C / f_M)
    let R_f = 14.0; // modulator 7x below carrier

    // carrier frequency
    let f_C = Arc::new(AtomicF32::new(440.0));

    // modulator frequency
    let f_M = Arc::new(AtomicF32::new(f_C.load(Ordering::Relaxed) / R_f));

    // modulator amplitude, calculated from I
    let A_M = Arc::new(AtomicF32::new(I * f_M.load(Ordering::Relaxed)));

    let modulator = WavetableOscillator::new(
        44100,
        wave_table.clone(),
        Arc::clone(&A_M),
        Arc::clone(&f_M),
        None, // modulator has no modulator
    );
    let mut carrier = WavetableOscillator::new(
        44100,
        wave_table.clone(),
        Arc::clone(&A_C),
        Arc::clone(&f_C),
        Some(Arc::new(Mutex::new(WavetableOscillator::new(
            44100,
            wave_table.clone(),
            Arc::new(AtomicF32::new(0.0)),
            Arc::new(AtomicF32::new(0.0)),
            None,
        )))), // carrier needs an input port for modulator
    );

    // Create the audio graph
    let mut graph = Graph::<Arc<Mutex<WavetableOscillator>>, ()>::new();

    // Add nodes to the graph
    let modulator_node = graph.add_node(Arc::new(Mutex::new(modulator)));
    let carrier_node = graph.add_node(Arc::new(Mutex::new(carrier)));

    // Connect modulator output to carrier input
    graph.add_edge(modulator_node, carrier_node, ());

    // For now, let's debug what's happening
    println!("Graph created with {} nodes", graph.node_count());
    println!("Modulator node: {:?}", modulator_node);
    println!("Carrier node: {:?}", carrier_node);

    // Create a simple connection by storing node indices
    // This will be used to transfer data during compute
    let connection = (modulator_node, carrier_node);

    println!(
        "Created connection from modulator {:?} to carrier {:?}",
        modulator_node, carrier_node
    );

    // Create a graph-based source for rodio
    struct GraphSource {
        graph: Graph<Arc<Mutex<WavetableOscillator>>, ()>,
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
                let node = &self.graph[*node_index];
                let mut node_guard = match node.lock() {
                    Ok(guard) => guard,
                    Err(_) => return None,
                };

                if let Err(_) = node_guard.compute() {
                    return None;
                }

                // Transfer data to connected nodes
                for (from_node, to_node) in &self.connections {
                    if from_node == node_index {
                        // Get the output data from current node
                        let output_data = if let Some(output) = node_guard.output(0) {
                            output.current_data()[0]
                        } else {
                            continue;
                        };

                        // Send to connected node
                        let to_node = &self.graph[*to_node];
                        let mut to_node_guard = to_node.lock().unwrap();
                        to_node_guard.inputs[0].receive(&[output_data]);
                    }
                }
            }

            // Return the final output (from carrier)
            let carrier = &self.graph[self.carrier_node];
            let carrier_guard = match carrier.lock() {
                Ok(guard) => guard,
                Err(_) => return None,
            };

            if let Some(output) = carrier_guard.output(0) {
                let sample = output.current_data()[0];
                Some(sample)
            } else {
                Some(0.0)
            }
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
        connections: vec![connection],
        carrier_node,
        sample_rate: 44100,
    };

    // render to wav using graph
    output_to_wav(
        &mut GraphSource {
            graph: graph_source.graph.clone(),
            sorted_nodes,
            connections: vec![connection],
            carrier_node,
            sample_rate: 44100,
        },
        "output.wav",
        Duration::from_secs(5),
    )
    .expect("Failed to write to wav");

    // // play audio
    // let Ok(stream_handle) = OutputStreamBuilder::open_default_stream() else {
    //     todo!()
    // };

    // stream_handle.mixer().add(graph_source);

    // // adjust parameters over time
    // for _i in 0..1000 {
    //     let mut current_freq = f_C.load(Ordering::Relaxed);
    //     current_freq += 1.0;
    //     f_C.store(current_freq, Ordering::Relaxed);
    //     f_M.store(current_freq / R_f, Ordering::Relaxed);
    //     A_M.store(I * f_M.load(Ordering::Relaxed), Ordering::Relaxed);
    //     std::thread::sleep(Duration::from_millis(5));
    // }
}
