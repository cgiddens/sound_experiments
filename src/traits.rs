/// Represents a single audio frame (sample) at a given time
pub type Sample = f32;

/// A simple audio node that can compute samples
pub trait AudioNode {
    /// Computes the next sample based on inputs and internal state
    fn compute(&mut self) -> Sample;

    /// Resets the node's internal state
    fn reset(&mut self);
}
