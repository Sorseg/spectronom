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
        let high = 22300.0;
        let x = (state.x_offset + u32(i) / (state.bins_count) / u32(2)) % state.canvas_size[0];
        let val_scale = log(high) - log(low);
        let scaling = 1.0 / val_scale * f32(state.canvas_size[1]);
        let red = (i32(i) / i32(state.bins_count)) % 2;
        let green = (i32(i) / i32(state.bins_count) + 1) % 2;

        // # Single point
        {
        var mid = f32(state.canvas_size[1]) - (log(data[i][0]) - log(low)) * scaling;
        
         var res = textureLoad(output, vec2(i32(x), i32(mid)));
            textureStore(
                output,
                vec2(i32(x), i32(mid)),
                res + vec4(
                    data[i][1] * f32(red),
                    data[i][1] * f32(green),
                    0.0,
                    0.9
                )
            );
        }

        // # MULTIPOINT
        // let width = 22050.0 / f32(state.bins_count);
        // var end = f32(state.canvas_size[1]) - (log(data[i][0] - width / 2.0) - log(low)) * scaling;
        // let start = f32(state.canvas_size[1]) - (log(data[i][0] + width / 2.0) - log(low)) * scaling;

        // let max_pix = 30.0;
        // let clamped_end = clamp(end, start + 2.0, start + max_pix);
        // for (var yy = start; yy <= clamped_end; yy += 1.0) {
        //     let scale = (clamped_end - start);
        //     // -0.5..0.5
        //     let phase = ((yy - start)/scale - 0.5);
        //     let brightness = 0.5 - abs(phase);
        //     let brightness_clamped = max(brightness,0.1);
        //     var res = textureLoad(output, vec2(i32(x), i32(yy)));
        //     textureStore(
        //         output,
        //         vec2(i32(x), i32(yy)),
        //         res + vec4(
        //             data[i][1] * f32(red) * brightness_clamped,
        //             data[i][1] * f32(green) * brightness_clamped,
        //             0.0,
        //             0.9
        //         )
        //     );
        // }
        
    }
}
