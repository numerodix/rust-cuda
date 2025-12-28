use cust::prelude::*;
use image::ExtendedColorType;
use std::error::Error;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    let fp = std::env::args().nth(1).expect("need image filepath");
    let img = image::open(&fp)?;

    let buf: Vec<f32> = img.as_bytes().iter().map(|&b| b as f32 / 255.0).collect();

    let width = img.width() as usize;
    let height = img.height() as usize;
    let buf_len = buf.len();

    println!("loaded image: {} with dim: {}x{} and {} bytes", fp, width, height, buf_len);

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let buf_gpu = buf.as_slice().as_dbuf()?;

    // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
    // cost of the copy, but you need to be careful not to read from the buffer.
    let mut blur = vec![0.1f32; buf_len];
    let blur_buf = blur.as_slice().as_dbuf()?;

    // retrieve the `vecadd` kernel from the module so we can calculate the right launch config.
    let to_blur = module.get_function("to_blur")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    // let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;
    let block_size = 32;

    let grid_size = (buf_len as u32).div_ceil(block_size);

    // println!("using {grid_size} blocks and {block_size} threads per block");

    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            to_blur<<<(grid_size, grid_size), (block_size, block_size), 0, stream>>>(
                buf_gpu.as_device_ptr(),
                buf_gpu.len(),
                blur_buf.as_device_ptr(),
                width as isize,
                height as isize,
            )
        )?;
    }

    stream.synchronize()?;

    // copy back the data from the GPU.
    blur_buf.copy_to(&mut blur)?;

    let blurred_bytes = blur
        .iter()
        .map(|f| {
            let v = (f * 255.0) as u8;
            [v]
        })
        .flatten()
        .collect::<Vec<u8>>();

    image::save_buffer(
        "blurred.png",
        &blurred_bytes.as_slice(),
        width as u32,
        height as u32,
        ExtendedColorType::Rgb8,
    )
    .expect("save blurred img");

    Ok(())
}
