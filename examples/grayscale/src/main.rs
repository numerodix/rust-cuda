use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;

/// How many numbers to generate and add together.
const NUMBERS_LEN: usize = 4;
const WIDTH: usize = NUMBERS_LEN.isqrt();
const HEIGHT: usize = NUMBERS_LEN.isqrt();

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // // generate our random vectors.
    let mut wyrand = WyRand::new();

    let mut red = vec![1.0f32; NUMBERS_LEN];
    wyrand.fill(&mut red);

    let mut green = vec![2.0f32; NUMBERS_LEN];
    wyrand.fill(&mut green);

    let mut blue = vec![3.0f32; NUMBERS_LEN];
    wyrand.fill(&mut blue);

    // let mut rhs = vec![0.0f32; NUMBERS_LEN];
    // wyrand.fill(&mut rhs);

    // let red = &[100.0];
    // let green = &[33.0];
    // let blue = &[150.0];

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
    let vecadd = module.get_function("vecadd")?;

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
            vecadd<<<(grid_size, grid_size), (block_size, block_size), 0, stream>>>(
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

    println!("red: {:?}", red);
    println!("green: {:?}", green);
    println!("blue: {:?}", blue);

    println!("gray: {:?}", gray);

    Ok(())
}
