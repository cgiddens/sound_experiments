use nih_plug::prelude::*;
use sound_experiments::FMSynth;

fn main() {
    nih_export_standalone::<FMSynth>();
}
