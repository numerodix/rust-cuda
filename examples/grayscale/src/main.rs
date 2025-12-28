use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;

/// How many numbers to generate and add together.
// const NUMBERS_LEN: usize = 4;
// const WIDTH: usize = NUMBERS_LEN.isqrt();
// const HEIGHT: usize = NUMBERS_LEN.isqrt();

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    let fp = std::env::args().nth(1).expect("need image filepath");
    let img = image::open(fp)?;

    let rgb_img = img.to_rgb32f();

    let WIDTH = img.width() as usize;
    let HEIGHT = img.height() as usize;
    let NUMBERS_LEN = WIDTH * HEIGHT;

    let raw_bytes = rgb_img.as_raw();

    let mut red = vec![];
    let mut green = vec![];
    let mut blue = vec![];

    for chunk in raw_bytes.chunks_exact(3) {
        red.push(chunk[0]);
        green.push(chunk[1]);
        blue.push(chunk[2]);
    }

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

    // allocate the GPU memory needed to house our numbers and copy them over.
    let red_gpu = red.as_slice().as_dbuf()?;
    let green_gpu = green.as_slice().as_dbuf()?;
    let blue_gpu = blue.as_slice().as_dbuf()?;

    // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
    // cost of the copy, but you need to be careful not to read from the buffer.
    let mut gray = vec![0.1f32; NUMBERS_LEN];
    let gray_buf = gray.as_slice().as_dbuf()?;

    // retrieve the `vecadd` kernel from the module so we can calculate the right launch config.
    let to_grayscale = module.get_function("to_grayscale")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    // let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;
    let block_size = 32;

    let grid_size = (NUMBERS_LEN as u32).div_ceil(block_size);

    // println!("using {grid_size} blocks and {block_size} threads per block");

    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            to_grayscale<<<(grid_size, grid_size), (block_size, block_size), 0, stream>>>(
                red_gpu.as_device_ptr(),
                red_gpu.len(),
                green_gpu.as_device_ptr(),
                green_gpu.len(),
                blue_gpu.as_device_ptr(),
                blue_gpu.len(),
                gray_buf.as_device_ptr(),
                WIDTH,
                HEIGHT,
            )
        )?;
    }

    stream.synchronize()?;

    // copy back the data from the GPU.
    gray_buf.copy_to(&mut gray)?;

    let gray_bytes = gray
        .iter()
        .map(|f| {
            let v = (f * 255.0) as u8;
            [v]
        })
        .flatten()
        .collect::<Vec<u8>>();

    let gray_img = image::GrayImage::from_raw(WIDTH as u32, HEIGHT as u32, gray_bytes)
        .expect("create gray img");
    gray_img.save("grayscale.png")?;

    // println!("red: {:?}", red);
    // println!("green: {:?}", green);
    // println!("blue: {:?}", blue);

    // println!("gray: {:?}", gray);

    Ok(())
}
