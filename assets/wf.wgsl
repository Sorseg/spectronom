@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<storage, read> data: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> state: WaterfallState;


struct WaterfallState {
    x_offset: u32,
    data_len: u32,
    canvas_size: vec2<u32>,
    bins_count: u32
}

// FIXME: parallelize
@compute @workgroup_size(1, 1, 1)
fn draw(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {

    // blank the target
    for (var xoff = u32(0); xoff < (state.data_len / state.bins_count / u32(2)); xoff++) {
        let x = (state.x_offset + xoff) % state.canvas_size[0];
        for (var y = u32(0); y < state.canvas_size[1]; y++) {
            textureStore(
                output,
                vec2<i32>(i32(x), i32(y)),
                vec4(0.0, 0.0, 0.0, 1.0)
            );
        }
    }

    // draw the frame
    for (var i = 0; i < i32(state.data_len); i++) {
        let low = 10.0;
        let high = 22050.0;
        let x = (state.x_offset + u32(i) / (state.bins_count) / u32(2)) % state.canvas_size[0];
        let val_scale = log(high) - log(low);
        // let y = f32(state.canvas_size[1]) - log(data[i][0]) / log(24100.0) * f32(state.canvas_size[1]);
        let red = (i32(i) / i32(state.bins_count)) % 2;
        let green = (i32(i) / i32(state.bins_count) + 1) % 2;
        // let width = 22050.0 / f32(state.bins_count);
        // var start = log(data[i][0] - width / 2.0 - low) / val_scale * f32(state.canvas_size[1]);
        // start = max(0.0, start);
        // let end = log(data[i][0] + width / 2.0 - low) /val_scale * f32(state.canvas_size[1]);
        var y = f32(state.canvas_size[1]) - (log(data[i][0]) - log(low)) / val_scale * f32(state.canvas_size[1]);
        
        let brightness = 1.0;
            
        var res = textureLoad(output, vec2(i32(x), i32(y)));
        textureStore(
            output,
            vec2(i32(x), i32(y)),
            res + vec4(
                    data[i][1] * f32(red) * brightness,
                    data[i][1] * f32(green) * brightness,
                    0.0,
                    0.9
                )
            );
        
    }
}
