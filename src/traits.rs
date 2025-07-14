use std::sync::Arc;
use std::sync::Mutex;

/// Represents a single audio frame (sample) at a given time
pub type Sample = f32;

/// Represents a buffer of audio samples for efficient batch processing
pub type SampleBuffer = Vec<Sample>;

/// Represents a single input or output connection
#[derive(Debug, Clone)]
pub struct Port {
    pub name: String,
    pub sample_rate: u32,
    pub buffer_size: usize, // 1 for single sample, >1 for buffer processing
}

/// Represents an input port that can receive audio data
pub trait InputPort: Send {
    /// Returns the port configuration
    fn port(&self) -> &Port;

    /// Receives a batch of input - buffer size is guaranteed to match at connection time
    fn receive(&mut self, data: &[Sample]);

    /// Returns the current data in the buffer
    fn current_data(&self) -> &[Sample];
}

/// Represents an output port that can send audio data
pub trait OutputPort: Send {
    /// Returns the port configuration
    fn port(&self) -> &Port;

    /// Sends data to all connected inputs - buffer size is guaranteed to match at connection time
    fn send(&mut self, data: &[Sample]);

    /// Returns the current data in the buffer
    fn current_data(&self) -> &[Sample];

    /// Connects this output to an input port
    fn connect_to(&mut self, input: Arc<Mutex<dyn InputPort + Send>>);

    /// Gets a mutable reference to the connected inputs (for direct manipulation)
    fn connected_inputs_mut(&mut self) -> &mut Vec<Arc<Mutex<dyn InputPort + Send>>>;

    /// Direct connection method for concrete types
    fn connect_to_concrete(&mut self, input: &mut ConcreteInputPort);
}

/// Error when trying to connect ports with mismatched buffer sizes
#[derive(Debug)]
pub enum ConnectionError {
    BufferSizeMismatch {
        from_size: usize,
        to_size: usize,
    },
    InvalidPortIndex {
        node_index: usize,
        port_index: usize,
    },
}

impl std::fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionError::BufferSizeMismatch { from_size, to_size } => {
                write!(
                    f,
                    "Cannot connect ports: buffer size mismatch ({} vs {})",
                    from_size, to_size
                )
            }
            ConnectionError::InvalidPortIndex {
                node_index,
                port_index,
            } => {
                write!(
                    f,
                    "Invalid port index: node {}, port {}",
                    node_index, port_index
                )
            }
        }
    }
}

impl std::error::Error for ConnectionError {}

/// A node in the audio processing graph
///
/// This trait represents any audio processing unit that can:
/// - Accept inputs from other nodes
/// - Produce outputs for other nodes
/// - Process audio data (compute/transform/apply mathematical operations)
///
/// The term "compute" is used instead of "get_sample" because it better reflects
/// the mathematical nature of audio processing - we're computing a value based
/// on inputs, internal state, and mathematical relationships.
pub trait AudioNode {
    /// Returns the sample rate this node operates at
    fn sample_rate(&self) -> u32;

    /// Returns the number of input ports
    fn input_count(&self) -> usize;

    /// Returns the number of output ports
    fn output_count(&self) -> usize;

    /// Gets a reference to an input port by index
    fn input(&self, index: usize) -> Option<&dyn InputPort>;

    /// Gets a mutable reference to an input port by index
    fn input_mut(&mut self, index: usize) -> Option<&mut dyn InputPort>;

    /// Gets a reference to an output port by index
    fn output(&self, index: usize) -> Option<&dyn OutputPort>;

    /// Gets a mutable reference to an output port by index
    fn output_mut(&mut self, index: usize) -> Option<&mut dyn OutputPort>;

    /// Computes the next output sample(s) based on current inputs and internal state
    ///
    /// This is the core mathematical operation of the node. For:
    /// - Oscillators: computes the next sample based on frequency, phase, etc.
    /// - Filters: applies transfer function to input samples
    /// - Effects: applies mathematical transformations (reverb, delay, etc.)
    /// - Mixers: combines multiple inputs mathematically
    fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Resets the node's internal state (phase, buffers, etc.)
    fn reset(&mut self);

    /// Returns true if the node is active/enabled
    fn is_active(&self) -> bool;

    /// Sets the active state of the node
    fn set_active(&mut self, active: bool);
}

/// A concrete implementation of an input port
#[derive(Clone, Debug)]
pub struct ConcreteInputPort {
    port: Port,
    current_sample: Sample,
    buffer: SampleBuffer,
}

impl ConcreteInputPort {
    pub fn new(name: String, sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            port: Port {
                name,
                sample_rate,
                buffer_size,
            },
            current_sample: 0.0,
            buffer: vec![0.0; buffer_size],
        }
    }
}

impl InputPort for ConcreteInputPort {
    fn port(&self) -> &Port {
        &self.port
    }
    fn receive(&mut self, data: &[Sample]) {
        // Buffer size is guaranteed to match at connection time
        // In debug builds, we can still assert for safety
        debug_assert_eq!(
            data.len(),
            self.port.buffer_size,
            "Buffer size mismatch: expected {}, got {}",
            self.port.buffer_size,
            data.len()
        );

        // Copy the data exactly
        self.buffer.copy_from_slice(data);

        // Update current sample (first element)
        if !data.is_empty() {
            self.current_sample = data[0];
        }
    }

    fn current_data(&self) -> &[Sample] {
        &self.buffer[..self.port.buffer_size]
    }
}

/// A concrete implementation of an output port
#[derive(Clone)]
pub struct ConcreteOutputPort {
    port: Port,
    current_sample: Sample,
    buffer: SampleBuffer,
    connected_inputs: Vec<Arc<Mutex<dyn InputPort + Send>>>,
}

/// Data stored on petgraph edges
pub struct EdgeData {
    pub from_port: usize,
    pub to_port: usize,
    pub buffer_size: Option<usize>, // None until validated
}

/// Manages audio connections and validation
pub struct ConnectionManager {
    nodes: Vec<Arc<Mutex<dyn AudioNode + Send>>>,
}

/// Represents a validated connection
pub struct AudioConnection {
    pub from_node: usize,
    pub from_port: usize,
    pub to_node: usize,
    pub to_port: usize,
    pub buffer_size: usize,
}

impl ConcreteOutputPort {
    pub fn new(name: String, sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            port: Port {
                name,
                sample_rate,
                buffer_size,
            },
            current_sample: 0.0,
            buffer: vec![0.0; buffer_size],
            connected_inputs: Vec::new(),
        }
    }

    pub fn connect_to(&mut self, input: Arc<Mutex<dyn InputPort + Send>>) {
        self.connected_inputs.push(input);
    }
}

impl ConnectionManager {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: Arc<Mutex<dyn AudioNode + Send>>) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(node);
        node_index
    }

    /// Validates an audio connection when petgraph creates an edge
    pub fn validate_connection(
        &self,
        from_node: usize,
        to_node: usize,
        edge_data: &mut EdgeData,
    ) -> Result<AudioConnection, ConnectionError> {
        // Validate node indices
        if from_node >= self.nodes.len() {
            return Err(ConnectionError::InvalidPortIndex {
                node_index: from_node,
                port_index: edge_data.from_port,
            });
        }
        if to_node >= self.nodes.len() {
            return Err(ConnectionError::InvalidPortIndex {
                node_index: to_node,
                port_index: edge_data.to_port,
            });
        }

        // Get the nodes
        let from_node_arc = &self.nodes[from_node];
        let to_node_arc = &self.nodes[to_node];

        // Lock the nodes to access their ports
        let from_node_guard =
            from_node_arc
                .lock()
                .map_err(|_| ConnectionError::InvalidPortIndex {
                    node_index: from_node,
                    port_index: edge_data.from_port,
                })?;
        let to_node_guard = to_node_arc
            .lock()
            .map_err(|_| ConnectionError::InvalidPortIndex {
                node_index: to_node,
                port_index: edge_data.to_port,
            })?;

        // Get the ports
        let from_output = from_node_guard.output(edge_data.from_port).ok_or_else(|| {
            ConnectionError::InvalidPortIndex {
                node_index: from_node,
                port_index: edge_data.from_port,
            }
        })?;
        let to_input = to_node_guard.input(edge_data.to_port).ok_or_else(|| {
            ConnectionError::InvalidPortIndex {
                node_index: to_node,
                port_index: edge_data.to_port,
            }
        })?;

        // Validate buffer sizes match
        if from_output.port().buffer_size != to_input.port().buffer_size {
            return Err(ConnectionError::BufferSizeMismatch {
                from_size: from_output.port().buffer_size,
                to_size: to_input.port().buffer_size,
            });
        }

        // Update the edge data with validated buffer size
        edge_data.buffer_size = Some(from_output.port().buffer_size);

        // Create the audio connection
        let audio_connection = AudioConnection {
            from_node,
            from_port: edge_data.from_port,
            to_node,
            to_port: edge_data.to_port,
            buffer_size: from_output.port().buffer_size,
        };

        Ok(audio_connection)
    }

    /// Gets a node by index
    pub fn get_node(&self, index: usize) -> Option<&Arc<Mutex<dyn AudioNode + Send>>> {
        self.nodes.get(index)
    }

    /// Establishes audio connections after validation
    /// This should be called after all edges are validated and added to petgraph
    pub fn establish_connections(
        &mut self,
        connections: &[AudioConnection],
    ) -> Result<(), ConnectionError> {
        for connection in connections {
            // Get the nodes
            let from_node = self.get_node(connection.from_node).ok_or_else(|| {
                ConnectionError::InvalidPortIndex {
                    node_index: connection.from_node,
                    port_index: connection.from_port,
                }
            })?;

            let to_node = self.get_node(connection.to_node).ok_or_else(|| {
                ConnectionError::InvalidPortIndex {
                    node_index: connection.to_node,
                    port_index: connection.to_port,
                }
            })?;

            // Lock the nodes
            let mut from_guard =
                from_node
                    .lock()
                    .map_err(|_| ConnectionError::InvalidPortIndex {
                        node_index: connection.from_node,
                        port_index: connection.from_port,
                    })?;

            let mut to_guard = to_node
                .lock()
                .map_err(|_| ConnectionError::InvalidPortIndex {
                    node_index: connection.to_node,
                    port_index: connection.to_port,
                })?;

            // Get the ports
            let from_output = from_guard.output_mut(connection.from_port).ok_or_else(|| {
                ConnectionError::InvalidPortIndex {
                    node_index: connection.from_node,
                    port_index: connection.from_port,
                }
            })?;

            let to_input = to_guard.input_mut(connection.to_port).ok_or_else(|| {
                ConnectionError::InvalidPortIndex {
                    node_index: connection.to_node,
                    port_index: connection.to_port,
                }
            })?;

            // Establish the connection
            // Note: This requires the concrete types to have a way to connect
            // We might need to add a method to the traits or use a different approach
            self.connect_ports(from_output, to_input)?;
        }

        Ok(())
    }

    /// Helper method to connect two ports
    fn connect_ports(
        &self,
        output: &mut dyn OutputPort,
        input: &mut dyn InputPort,
    ) -> Result<(), ConnectionError> {
        // This is where we'd actually establish the connection
        // For now, we'll need to think about the best way to do this
        // The issue is that we need to cast to concrete types to access
        // the connection methods, but we're working with trait objects

        // One approach: Add a connect method to the traits
        // Another approach: Use a different architecture for connections

        Ok(())
    }
}

impl OutputPort for ConcreteOutputPort {
    fn port(&self) -> &Port {
        &self.port
    }

    fn send(&mut self, data: &[Sample]) {
        // Update our internal buffer first
        self.buffer.copy_from_slice(data);
        if !data.is_empty() {
            self.current_sample = data[0];
        }

        // Send to all connected inputs
        for input in &self.connected_inputs {
            if let Ok(mut input_guard) = input.lock() {
                input_guard.receive(data);
            }
        }
    }

    fn current_data(&self) -> &[Sample] {
        &self.buffer[..self.port.buffer_size]
    }

    fn connect_to(&mut self, input: Arc<Mutex<dyn InputPort + Send>>) {
        self.connected_inputs.push(input);
    }

    fn connected_inputs_mut(&mut self) -> &mut Vec<Arc<Mutex<dyn InputPort + Send>>> {
        &mut self.connected_inputs
    }

    /// Direct connection method for concrete types
    fn connect_to_concrete(&mut self, input: &mut ConcreteInputPort) {
        // For now, we'll use a simple approach: store a reference to the input
        // This is a temporary solution until we fix the proper connection system
    }
}
