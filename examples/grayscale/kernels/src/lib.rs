use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(red: &[f32], green: &[f32], blue: &[f32], gray: *mut f32, width: usize, height: usize) {
    let row = (thread::block_idx_y() * thread::block_dim_y() + thread::thread_idx_y()) as usize;
    let col = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;

    if row >= height || col >= width {
        return;
    }

    let idx = row * width + col;

    let red_cell = red[idx];
    let green_cell = green[idx];
    let blue_cell = blue[idx];

    let gray_cell = unsafe { &mut *gray.add(idx) };

    *gray_cell = 0.3 * red_cell + 0.6 * green_cell + 0.1 * blue_cell;
}
