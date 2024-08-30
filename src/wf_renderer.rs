use std::borrow::Cow;

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{storage_buffer_read_only, texture_storage_2d, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
            BufferVec, CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, PipelineCache, ShaderStages, ShaderType,
            StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSet,
    },
    window::PrimaryWindow,
};

use crate::{FftReceiver, CHANNELS_TO_DISPLAY, FFT_BINS};

const WF_HEIGHT: u32 = 1080 * 2;
const WF_WIDTH: u32 = 1920 * 2;

#[derive(Resource, Clone, Default, ExtractResource, ShaderType)]
struct WaterfallState {
    x_offset: u32,
    data_len: u32,
    canvas_size: UVec2,
    bins_count: u32,
}

#[derive(Resource)]
struct WaterfallBuffers {
    values: BufferVec<Vec2>,
    state: UniformBuffer<WaterfallState>,
}

impl FromWorld for WaterfallBuffers {
    fn from_world(_world: &mut World) -> Self {
        WaterfallBuffers {
            values: BufferVec::new(BufferUsages::COPY_DST | BufferUsages::STORAGE),
            state: UniformBuffer::default(),
        }
    }
}

#[derive(Resource)]
struct WfPipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for WfPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            "WfImage",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::ReadWrite),
                    storage_buffer_read_only::<Vec2>(false),
                    uniform_buffer::<WaterfallState>(false),
                ),
            ),
        );
        let shader = world.load_asset("wf.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("draw"),
        });

        WfPipeline {
            layout,
            pipeline: init_pipeline,
        }
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct WfImage(Handle<Image>);

pub(crate) struct WfRenderPlugin;

#[derive(Component)]
pub(crate) struct WaterfallSprite;

impl Plugin for WfRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaterfallState>()
            .add_plugins(ExtractResourcePlugin::<WfImage>::default())
            .add_plugins(ExtractResourcePlugin::<FftReceiver>::default())
            .add_plugins(ExtractResourcePlugin::<WaterfallState>::default())
            .add_systems(Startup, setup)
            .add_systems(Update, stretch_spectrogram);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            (
                receive_bins.in_set(RenderSet::Prepare),
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
            ),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();

        render_graph.add_node(WaterfallLabel, WaterfallNode::default());
        render_graph.add_node_edge(WaterfallLabel, bevy::render::graph::CameraDriverLabel);
    }
    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<WfPipeline>()
            .init_resource::<WaterfallBuffers>();
    }
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = Extent3d {
        width: WF_WIDTH,
        height: WF_HEIGHT,
        ..default()
    };

    // This is the texture that will be rendered to.
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(size);
    let image = images.add(image);
    commands.spawn((
        WaterfallSprite,
        SpriteBundle {
            texture: image.clone(),
            ..default()
        },
    ));
    commands.spawn(Camera2dBundle::default());
    commands.insert_resource(WfImage(image));
}

#[derive(Resource)]
struct WfBindGroup(BindGroup);

fn receive_bins(
    receiver: Res<FftReceiver>,
    mut buffers: ResMut<WaterfallBuffers>,
    mut state: ResMut<WaterfallState>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    state.bins_count = FFT_BINS as u32;
    state.x_offset += state.data_len / state.bins_count / CHANNELS_TO_DISPLAY as u32;
    state.x_offset %= WF_WIDTH;
    state.canvas_size = UVec2::new(WF_WIDTH, WF_HEIGHT);

    buffers.values.clear();
    let mut i = 0;
    while let Ok(v) = receiver.0.try_recv() {
        for v in v {
            buffers.values.push(Vec2::new(v.0, v.1));
            i += 1;
        }
    }
    state.data_len = i;
    buffers.state.set(state.clone());

    buffers.values.write_buffer(&device, &queue);
    buffers.state.write_buffer(&device, &queue);
}

// FIXME: the binding should not be recreated on each frame?
fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<WfPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    img: Res<WfImage>,
    render_device: Res<RenderDevice>,
    buffers: Res<WaterfallBuffers>,
) {
    let view = gpu_images.get(&img.0).unwrap();

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::sequential((
            &view.texture_view,
            buffers.values.binding().unwrap(),
            buffers.state.binding().unwrap(),
        )),
    );

    commands.insert_resource(WfBindGroup(bind_group));
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterfallLabel;

enum WaterfallShaderState {
    Loading,
    Working,
}

struct WaterfallNode {
    state: WaterfallShaderState,
}

impl Default for WaterfallNode {
    fn default() -> Self {
        Self {
            state: WaterfallShaderState::Loading,
        }
    }
}

impl render_graph::Node for WaterfallNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<WfPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            WaterfallShaderState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = WaterfallShaderState::Working;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing waterfall shader error:\n{err}")
                    }
                    _ => {}
                }
            }
            WaterfallShaderState::Working => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_group = &world.resource::<WfBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<WfPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.state {
            WaterfallShaderState::Loading => {}
            WaterfallShaderState::Working => {
                if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
        }

        Ok(())
    }
}

fn stretch_spectrogram(
    mut s: Query<&mut Sprite, With<WaterfallSprite>>,
    w: Query<&Window, With<PrimaryWindow>>,
) {
    if let Ok(w) = w.get_single() {
        let mut spr = s.single_mut();
        spr.custom_size = Some(w.resolution.size());
    }
}
