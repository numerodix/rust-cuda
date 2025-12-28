use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn to_blur(img: &[f32], blurred: *mut f32, width: isize, height: isize) {
    let out_row = (thread::block_idx_y() * thread::block_dim_y() + thread::thread_idx_y()) as isize;
    let out_col = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as isize;

    if out_row >= height || out_col >= width {
        return;
    }

    let idx = (out_row * width + out_col) as usize;
    let blurred_cell = unsafe { &mut *blurred.add(idx) };

    *blurred_cell = img[idx];

    return;

    let radius = 1;
    let mut avg = 0.0;

    for in_row in (out_row - radius)..=(out_row + radius) {
        for in_col in (out_col - radius)..=(out_col + radius) {
            if in_row < 0 || in_row >= height || in_col < 0 || in_col >= width {
                continue;
            }

            let idx = (in_row * width + in_col) as usize;
            avg += img[idx];
        }
    }

    let idx = (out_row * width + out_col) as usize;
    let blurred_cell = unsafe { &mut *blurred.add(idx) };

    let num_blurred_pixels = {
        let mut cnt = 2 * radius + 1;
        cnt *= cnt;
        cnt
    };

    *blurred_cell = avg / num_blurred_pixels as f32;
}
