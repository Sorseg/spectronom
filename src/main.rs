mod wf_renderer;

use std::{
    array,
    collections::VecDeque,
    f32::consts::{PI, TAU},
    time::Duration,
};

use bevy::{prelude::*, render::extract_resource::ExtractResource};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamConfig,
};
use crossbeam_channel::{bounded, Receiver};
use wf_renderer::WfRenderPlugin;

fn main() {
    let (sender, receiver) = bounded(1024);
    // FIXME: separate stream manager thread
    std::mem::forget(start_listener(sender));
    App::new()
        .add_plugins((
            DefaultPlugins
                // .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: bevy::window::PresentMode::AutoVsync,
                        mode: bevy::window::WindowMode::BorderlessFullscreen,
                        ..default()
                    }),
                    ..default()
                }),
            WfRenderPlugin,
        ))
        .insert_resource(FftReceiver(receiver))
        .add_systems(Update, quit_on_esc)
        .run();
}

const WINDOW_SIZE: usize = 1024 * 4;
const FFT_BINS: usize = WINDOW_SIZE / 2;
const CHANNELS_TO_DISPLAY: usize = 2;

#[derive(Clone)]
struct ChannelAnalysis {
    gathered_sound: VecDeque<f32>,
    phases: Vec<f32>,
    noise: Vec<f64>,
    analyzed: VecDeque<[(f32, f32); FFT_BINS]>,
}

impl Default for ChannelAnalysis {
    fn default() -> Self {
        Self {
            gathered_sound: VecDeque::with_capacity(WINDOW_SIZE),
            phases: vec![0.0; FFT_BINS],
            noise: vec![0.0; FFT_BINS],
            analyzed: VecDeque::default(),
        }
    }
}

fn start_listener(
    sender: crossbeam_channel::Sender<[(f32, f32); FFT_BINS * CHANNELS_TO_DISPLAY]>,
) -> cpal::Stream {
    println!("Available hosts {:#?}", cpal::available_hosts());
    let dev = cpal::default_host().default_input_device().unwrap();
    let config: StreamConfig = dev.default_input_config().unwrap().into();

    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0;

    let mut analysis_per_chan = vec![ChannelAnalysis::default(); channels];

    println!("Creating stream for {:?}: {:?}", dev.name(), config);

    let mut buf = [0.0; WINDOW_SIZE];

    let offset = 128;
    let noise_detection_length = sample_rate as usize / offset * 10;
    let bin_width = sample_rate as f32 / WINDOW_SIZE as f32;
    let dt = offset as f32 / sample_rate as f32;
    let hann: [f32; WINDOW_SIZE] =
        array::from_fn(|i| (i as f32 * PI / WINDOW_SIZE as f32).sin().powi(2));
    let s = dev
        .build_input_stream(
            &config,
            move |d: &[f32], _| {
                for chunk in d.chunks(channels) {
                    for (i, val) in chunk.iter().copied().enumerate() {
                        analysis_per_chan[i].gathered_sound.push_back(val);
                    }
                }
                for chan in analysis_per_chan.iter_mut() {
                    while chan.gathered_sound.len() >= WINDOW_SIZE {
                        for (i, (target, window)) in buf.iter_mut().zip(&hann).enumerate() {
                            *target = if i < offset {
                                chan.gathered_sound.pop_front().unwrap()
                            } else {
                                *chan.gathered_sound.get(i - offset).unwrap()
                            };
                            *target *= window;
                        }
                        let res = microfft::real::rfft_4096(&mut buf);

                        let mut analyzed = [(0.0, 0.0); FFT_BINS];
                        for (i, (fft_bin, target)) in res.iter().zip(&mut analyzed).enumerate() {
                            let (mag, phase) = fft_bin.to_polar();
                            let phase = (phase + TAU) % TAU;
                            let phase_diff = (phase - chan.phases[i]) % PI;
                            let bin_freq = bin_width * i as f32;
                            let mut expected_phase_diff = (dt * bin_freq * TAU) % TAU;
                            if expected_phase_diff > PI {
                                expected_phase_diff -= TAU;
                            }
                            let phase_error = (phase_diff - expected_phase_diff) % PI;
                            let freq_deviation = phase_error / TAU / dt;
                            chan.noise[i] = (chan.noise[i] * (noise_detection_length - 1) as f64
                                + mag as f64)
                                / noise_detection_length as f64;

                            target.0 =
                                (bin_freq + freq_deviation).clamp(0.0, sample_rate as f32 / 2.0);
                            target.1 = (mag - chan.noise[i] as f32 * 0.7).clamp(0.0, 3.0);

                            chan.phases[i] = phase;
                        }
                        chan.analyzed.push_back(analyzed);
                    }
                }
                // FIXME: there has to be a better way
                while !analysis_per_chan[0].analyzed.is_empty() {
                    let mut result = [(0.0, 0.0); FFT_BINS * CHANNELS_TO_DISPLAY];
                    for (i, chan) in analysis_per_chan
                        .iter_mut()
                        .enumerate()
                        .take(CHANNELS_TO_DISPLAY)
                    {
                        for (ii, val) in chan.analyzed.pop_front().unwrap().into_iter().enumerate()
                        {
                            result[i * FFT_BINS + ii] = val;
                        }
                    }
                    sender.try_send(result).unwrap();
                }
                
            },
            |e| eprintln!("{e:?}"),
            Some(Duration::from_secs(3)),
        )
        .unwrap();
    s.play().unwrap();
    s
}

#[derive(Resource, Clone, ExtractResource)]
struct FftReceiver(Receiver<[(f32, f32); FFT_BINS * CHANNELS_TO_DISPLAY]>);

fn quit_on_esc(input: Res<ButtonInput<KeyCode>>, mut exit_event: EventWriter<AppExit>) {
    if input.just_pressed(KeyCode::Escape) {
        exit_event.send(AppExit::Success);
    }
}
