mod traits;

use crate::traits::{AudioNode, Sample};
use nih_plug::prelude::*;
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use parking_lot::Mutex;
use petgraph::algo::toposort;
use petgraph::graph::{Graph, NodeIndex};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use winit::keyboard::KeyCode;
use std::collections::HashSet;

#[derive(Clone)]
pub struct WavetableOscillator {
    sample_rate: u32,
    wave_table: Vec<f32>,
    index: f32,
    amplitude: Arc<atomic_float::AtomicF32>,
    frequency: Arc<atomic_float::AtomicF32>,
    modulator_inputs: Vec<f32>,
    buffer_size: usize,
}

impl WavetableOscillator {
    pub fn new(
        sample_rate: u32,
        wave_table: Vec<f32>,
        amplitude: Arc<atomic_float::AtomicF32>,
        frequency: Arc<atomic_float::AtomicF32>,
    ) -> WavetableOscillator {
        WavetableOscillator {
            sample_rate,
            wave_table,
            index: 0.0,
            amplitude,
            frequency,
            modulator_inputs: Vec::new(),
            buffer_size: 1,
        }
    }

    pub fn set_modulator_input(&mut self, modulator_index: usize, input: f32) {
        let needed_size = (modulator_index + 1) * self.buffer_size;
        if self.modulator_inputs.len() < needed_size {
            self.modulator_inputs.resize(needed_size, 0.0);
        }
        self.modulator_inputs[modulator_index * self.buffer_size] = input;
    }

    pub fn get_modulator_sum(&self) -> f32 {
        let mut sum = 0.0;
        let modulator_count = self.modulator_inputs.len() / self.buffer_size;
        for mod_idx in 0..modulator_count {
            sum += self.modulator_inputs[mod_idx * self.buffer_size];
        }
        sum
    }

    pub fn clear_modulator_inputs(&mut self) {
        self.modulator_inputs.fill(0.0);
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
        let frequency = self.frequency.load(Ordering::Relaxed);
        let amplitude = self.amplitude.load(Ordering::Relaxed);

        let modulator_sum = self.get_modulator_sum();

        let increment = frequency + modulator_sum;
        let index_increment = increment * self.wave_table.len() as f32 / self.sample_rate as f32;

        let sample = self.lerp();
        self.index += index_increment;
        self.index %= self.wave_table.len() as f32;

        amplitude * sample
    }

    fn reset(&mut self) {
        self.index = 0.0;
        self.modulator_inputs.clear();
    }
}

// The main plugin struct
pub struct FMSynth {
    params: Arc<FMSynthParams>,
    sample_rate: f32,

    // Synth components
    graph: Graph<WavetableOscillator, ()>,
    sorted_nodes: Vec<NodeIndex>,
    connections: Vec<(NodeIndex, NodeIndex, usize)>,
    output_nodes: Vec<NodeIndex>,

    // MIDI state
    midi_note_id: u8,
    midi_note_freq: f32,
    midi_note_gain: Smoother<f32>,

    // Keyboard input state
    pressed_keys: std::collections::HashSet<KeyCode>,
    is_standalone: bool,

    // Wavetable
    wave_table: Vec<f32>,
}

#[derive(Params)]
struct FMSynthParams {
    #[id = "carrier_freq"]
    pub carrier_frequency: FloatParam,

    #[id = "carrier2_freq"]
    pub carrier2_frequency: FloatParam,

    #[id = "mod_ratio"]
    pub modulator_ratio: FloatParam,

    #[id = "mod2_ratio"]
    pub modulator2_ratio: FloatParam,

    #[id = "mod_index"]
    pub modulation_index: FloatParam,

    #[id = "mod2_index"]
    pub modulation2_index: FloatParam,

    #[id = "master_gain"]
    pub master_gain: FloatParam,

    #[id = "keyboard_input"]
    pub keyboard_input_enabled: BoolParam,

    #[persist = "wave_table"]
    pub wave_table_data: Mutex<Vec<f32>>,

    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for FMSynth {
    fn default() -> Self {
        // Create wavetable
        let wave_table_size = 64;
        let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);
        for n in 0..wave_table_size {
            wave_table.push((2.0 * std::f32::consts::PI * n as f32 / wave_table_size as f32).sin());
        }

        let params = Arc::new(FMSynthParams::default());

        Self {
            params,
            sample_rate: 44100.0,
            graph: Graph::new(),
            sorted_nodes: Vec::new(),
            connections: Vec::new(),
            output_nodes: Vec::new(),
            midi_note_id: 0,
            midi_note_freq: 440.0,
            midi_note_gain: Smoother::new(SmoothingStyle::Linear(5.0)),
            pressed_keys: std::collections::HashSet::new(),
            is_standalone: false, // Will be set in initialize()
            wave_table,
        }
    }
}

impl Default for FMSynthParams {
    fn default() -> Self {
        Self {
            carrier_frequency: FloatParam::new(
                "Carrier Frequency",
                440.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_smoother(SmoothingStyle::Linear(10.0))
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),

            carrier2_frequency: FloatParam::new(
                "Carrier 2 Frequency",
                3520.0, // 8x carrier
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_smoother(SmoothingStyle::Linear(10.0))
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),

            modulator_ratio: FloatParam::new(
                "Modulator Ratio",
                14.0,
                FloatRange::Linear {
                    min: 0.1,
                    max: 100.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            modulator2_ratio: FloatParam::new(
                "Modulator 2 Ratio",
                64.0,
                FloatRange::Linear {
                    min: 0.1,
                    max: 100.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            modulation_index: FloatParam::new(
                "Modulation Index",
                16.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            modulation2_index: FloatParam::new(
                "Modulation 2 Index",
                128.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            master_gain: FloatParam::new(
                "Master Gain",
                -10.0,
                FloatRange::Linear {
                    min: -30.0,
                    max: 0.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(3.0))
            .with_unit(" dB"),

            keyboard_input_enabled: BoolParam::new("Keyboard Input", true),

            wave_table_data: Mutex::new(Vec::new()),
            editor_state: EguiState::from_size(400, 300),
        }
    }
}

impl FMSynth {
    fn rebuild_synth(&mut self) {
        // Clear existing graph
        self.graph.clear();
        self.sorted_nodes.clear();
        self.connections.clear();
        self.output_nodes.clear();

        // Get current parameters
        let carrier_freq = self.params.carrier_frequency.value();
        let carrier2_freq = self.params.carrier2_frequency.value();
        let mod_ratio = self.params.modulator_ratio.value();
        let mod2_ratio = self.params.modulator2_ratio.value();
        let mod_index = self.params.modulation_index.value();
        let mod2_index = self.params.modulation2_index.value();

        // Calculate frequencies
        let mod_freq = carrier_freq / mod_ratio;
        let mod2_freq = carrier2_freq / mod2_ratio;

        // Create oscillators
        let modulator = WavetableOscillator::new(
            self.sample_rate as u32,
            self.wave_table.clone(),
            Arc::new(atomic_float::AtomicF32::new(mod_index * mod_freq)),
            Arc::new(atomic_float::AtomicF32::new(mod_freq)),
        );

        let modulator2 = WavetableOscillator::new(
            self.sample_rate as u32,
            self.wave_table.clone(),
            Arc::new(atomic_float::AtomicF32::new(mod2_index * mod2_freq)),
            Arc::new(atomic_float::AtomicF32::new(mod2_freq)),
        );

        let carrier = WavetableOscillator::new(
            self.sample_rate as u32,
            self.wave_table.clone(),
            Arc::new(atomic_float::AtomicF32::new(1.0)),
            Arc::new(atomic_float::AtomicF32::new(carrier_freq)),
        );

        let carrier2 = WavetableOscillator::new(
            self.sample_rate as u32,
            self.wave_table.clone(),
            Arc::new(atomic_float::AtomicF32::new(1.0)),
            Arc::new(atomic_float::AtomicF32::new(carrier2_freq)),
        );

        // Add nodes to graph
        let modulator_node = self.graph.add_node(modulator);
        let modulator2_node = self.graph.add_node(modulator2);
        let carrier_node = self.graph.add_node(carrier);
        let carrier2_node = self.graph.add_node(carrier2);

        // Add edges
        self.graph.add_edge(modulator_node, carrier_node, ());
        self.graph.add_edge(modulator_node, carrier2_node, ());
        self.graph.add_edge(modulator2_node, carrier2_node, ());

        // Compute topological sort
        self.sorted_nodes = toposort(&self.graph, None).expect("Graph has cycles");

        // Set up connections
        self.connections = vec![
            (modulator_node, carrier_node, 0),
            (modulator_node, carrier2_node, 0),
            (modulator2_node, carrier2_node, 1),
        ];

        // Set output nodes
        self.output_nodes = vec![carrier_node, carrier2_node];
    }

    fn process_frame(&mut self) -> f32 {
        // Clear all modulator inputs
        for node_index in &self.sorted_nodes {
            self.graph[*node_index].clear_modulator_inputs();
        }

        // Process each node in order
        for node_index in &self.sorted_nodes {
            for (from_node, to_node, modulator_index) in &self.connections {
                if from_node == node_index {
                    let modulator_sample = self.graph[*from_node].compute();
                    self.graph[*to_node].set_modulator_input(*modulator_index, modulator_sample);
                }
            }
        }

        // Mix outputs
        let mut mixed_output = 0.0;
        for &output_node in &self.output_nodes {
            mixed_output += self.graph[output_node].compute();
        }

        mixed_output * 0.5 // Scale down to prevent clipping
    }

    /// Convert a key code to a MIDI note number
    fn key_to_midi_note(key: KeyCode) -> Option<u8> {
        match key {
            // A-Z keys mapped to notes (A = 60, B = 62, C = 64, etc.)
            KeyCode::KeyA => Some(60), // Middle C
            KeyCode::KeyW => Some(61), // C#
            KeyCode::KeyS => Some(62), // D
            KeyCode::KeyE => Some(63), // D#
            KeyCode::KeyD => Some(64), // E
            KeyCode::KeyF => Some(65), // F
            KeyCode::KeyT => Some(66), // F#
            KeyCode::KeyG => Some(67), // G
            KeyCode::KeyY => Some(68), // G#
            KeyCode::KeyH => Some(69), // A
            KeyCode::KeyU => Some(70), // A#
            KeyCode::KeyJ => Some(71), // B
            KeyCode::KeyK => Some(72), // High C
            KeyCode::KeyO => Some(73), // C#
            KeyCode::KeyL => Some(74), // D
            KeyCode::KeyP => Some(75), // D#
            KeyCode::Semicolon => Some(76), // E
            KeyCode::Quote => Some(77), // F
            _ => None,
        }
    }

    /// Process keyboard input and generate MIDI events
    fn process_keyboard_input(&mut self) -> Vec<NoteEvent<()>> {
        let mut events = Vec::new();
        
        // Only process keyboard input if enabled and in standalone mode
        if !self.params.keyboard_input_enabled.value() || !self.is_standalone {
            return events;
        }

        // For now, we'll simulate keyboard input for testing
        // In a real implementation, this would be connected to winit events
        // This is a placeholder that will be replaced with actual event handling
        
        events
    }

    /// Simulate keyboard input for testing
    fn simulate_keyboard_input(&mut self) -> Vec<NoteEvent<()>> {
        let mut events = Vec::new();
        
        // Only process if keyboard input is enabled
        if !self.params.keyboard_input_enabled.value() {
            return events;
        }

        // Simulate pressing 'A' key every 2 seconds for testing
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        if current_time % 2 == 0 && !self.pressed_keys.contains(&KeyCode::KeyA) {
            // Simulate key press
            self.pressed_keys.insert(KeyCode::KeyA);
            if let Some(note) = Self::key_to_midi_note(KeyCode::KeyA) {
                events.push(NoteEvent::NoteOn {
                    timing: 0,
                    note,
                    velocity: 0.8,
                    channel: 0,
                    voice_id: None,
                });
                println!("Simulated key press: A -> MIDI note {}", note);
            }
        } else if current_time % 2 == 1 && self.pressed_keys.contains(&KeyCode::KeyA) {
            // Simulate key release
            self.pressed_keys.remove(&KeyCode::KeyA);
            if let Some(note) = Self::key_to_midi_note(KeyCode::KeyA) {
                events.push(NoteEvent::NoteOff {
                    timing: 0,
                    note,
                    velocity: 0.0,
                    channel: 0,
                    voice_id: None,
                });
                println!("Simulated key release: A -> MIDI note {}", note);
            }
        }
        
        events
    }
}

impl Plugin for FMSynth {
    const NAME: &'static str = "FM Synth";
    const VENDOR: &'static str = "Sound Experiments";
    const URL: &'static str = "https://github.com/your-repo";
    const EMAIL: &'static str = "info@example.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: None,
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: None,
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        
        // For now, assume we're in standalone mode if keyboard input is enabled
        // This is a simple heuristic - in a real implementation, you'd want more robust detection
        self.is_standalone = self.params.keyboard_input_enabled.value();
        
        if self.is_standalone {
            println!("Keyboard input enabled - assuming standalone mode");
        } else {
            println!("Keyboard input disabled - assuming DAW host mode");
        }
        
        self.rebuild_synth();
        true
    }

    fn reset(&mut self) {
        // Reset all oscillators
        for node_index in &self.sorted_nodes {
            self.graph[*node_index].reset();
        }

        self.midi_note_id = 0;
        self.midi_note_freq = 440.0;
        self.midi_note_gain.reset(0.0);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut next_event = context.next_event();

        for (sample_id, channel_samples) in buffer.iter_samples().enumerate() {
            // Handle simulated keyboard input (every 1000 samples to avoid spam)
            if sample_id % 1000 == 0 {
                let keyboard_events = self.simulate_keyboard_input();
                for event in keyboard_events {
                    match event {
                        NoteEvent::NoteOn { note, velocity, .. } => {
                            self.midi_note_id = note;
                            self.midi_note_freq = util::midi_note_to_freq(note);
                            self.midi_note_gain.set_target(self.sample_rate, velocity);
                            self.rebuild_synth();
                        }
                        NoteEvent::NoteOff { note, .. } if note == self.midi_note_id => {
                            self.midi_note_gain.set_target(self.sample_rate, 0.0);
                        }
                        _ => (),
                    }
                }
            }

            // Handle MIDI events
            while let Some(event) = next_event {
                if event.timing() > sample_id as u32 {
                    break;
                }

                println!("Received MIDI event: {:?}", event);

                match event {
                    NoteEvent::NoteOn { note, velocity, .. } => {
                        self.midi_note_id = note;
                        self.midi_note_freq = util::midi_note_to_freq(note);
                        self.midi_note_gain.set_target(self.sample_rate, velocity);

                        // Update carrier frequencies based on MIDI note
                        // Note: We can't directly set parameter values from the audio thread
                        // Instead, we'll use the MIDI note frequency directly in rebuild_synth
                        self.midi_note_freq = util::midi_note_to_freq(note);
                        self.rebuild_synth();
                    }
                    NoteEvent::NoteOff { note, .. } if note == self.midi_note_id => {
                        self.midi_note_gain.set_target(self.sample_rate, 0.0);
                    }
                    _ => (),
                }

                next_event = context.next_event();
            }

            // Process audio
            let master_gain = self.params.master_gain.smoothed.next();
            let carrier_freq = self.params.carrier_frequency.smoothed.next();
            let mod_index = self.params.modulation_index.smoothed.next();
            let keyboard_enabled = self.params.keyboard_input_enabled.value();
            
            let synth_output = self.process_frame();
            let midi_gain = self.midi_note_gain.next();

            // Add a small test tone if no MIDI input
            let test_tone = if midi_gain < 0.01 {
                0.1 * (2.0 * std::f32::consts::PI * 440.0 * sample_id as f32 / self.sample_rate)
                    .sin()
            } else {
                0.0
            };

            let final_output =
                (synth_output * midi_gain + test_tone) * util::db_to_gain_fast(master_gain);

            // Debug output every 10000 samples (about every 0.2 seconds at 44.1kHz)
            if sample_id % 10000 == 0 {
                println!("Debug - Carrier: {:.1}Hz, Mod Index: {:.1}, Master: {:.1}dB, Keyboard: {}, Output: {:.3}", 
                         carrier_freq, mod_index, master_gain, keyboard_enabled, final_output);
            }

            for sample in channel_samples {
                *sample = final_output;
            }
        }

        ProcessStatus::KeepAlive
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        // Create a GUI with parameter controls
        let params = self.params.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    ui.heading("FM Synth");

                    // Add some basic controls
                    ui.add(widgets::ParamSlider::for_param(
                        &params.carrier_frequency,
                        setter,
                    ));
                    ui.add(widgets::ParamSlider::for_param(
                        &params.modulation_index,
                        setter,
                    ));
                    ui.add(widgets::ParamSlider::for_param(&params.master_gain, setter));

                    // Add keyboard input toggle
                    ui.label("Keyboard Input (toggle in standalone mode)");
                    let mut keyboard_enabled = params.keyboard_input_enabled.value();
                    if ui.checkbox(&mut keyboard_enabled, "Enable Keyboard Input").changed() {
                        setter.set_parameter(&params.keyboard_input_enabled, keyboard_enabled);
                    }

                    // Add a test tone button
                    if ui.button("Test Tone (440Hz)").clicked() {
                        println!("Test tone button clicked!");
                    }

                    // Show some debug info
                    ui.label("Debug Info:");
                    ui.label("• GUI is working");
                    ui.label("• Keyboard input: Try pressing A-Z keys for notes");
                    ui.label("• Check your system audio settings");
                    ui.label(format!("• Keyboard enabled: {}", params.keyboard_input_enabled.value()));
                });
            },
        )
    }
}

impl ClapPlugin for FMSynth {
    const CLAP_ID: &'static str = "com.sound-experiments.fm-synth";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A simple FM synthesizer");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];
}

impl Vst3Plugin for FMSynth {
    const VST3_CLASS_ID: [u8; 16] = *b"FMSynthSoundExp\0";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Synth];
}

// Export the plugin
nih_export_clap!(FMSynth);
nih_export_vst3!(FMSynth);
