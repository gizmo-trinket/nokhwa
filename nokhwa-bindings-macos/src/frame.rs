use crate::core_media::{
    CMSampleBufferGetImageBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetPixelFormatType,
    CVPixelBufferLockBaseAddress, CVPixelBufferRef, CVPixelBufferUnlockBaseAddress,
};
use core_media_sys::{kCMPixelFormat_422YpCbCr8_yuvs, CMSampleBufferRef};
use core_video_sys::{kCVPixelBufferLock_ReadOnly, kCVPixelFormatType_32BGRA};
use nokhwa_core::{
    error::NokhwaError,
    types::{FrameFormat, Resolution},
};
use std::slice;

const BGRA_BYTES_PER_PIXEL: usize = 4;
const RGB_BYTES_PER_PIXEL: usize = 3;
const YUYV_BYTES_PER_PIXEL: usize = 2;

#[allow(non_snake_case)]
#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferGetWidth(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetBytesPerRow(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetPlaneCount(buffer: CVPixelBufferRef) -> usize;
}

#[derive(Debug)]
pub struct DeliveredFrame {
    pub data: Vec<u8>,
    pub format: FrameFormat,
    pub resolution: Resolution,
}

fn error(message: impl Into<String>) -> NokhwaError {
    NokhwaError::ProcessFrameError {
        src: FrameFormat::NV12,
        destination: "RGB8".to_string(),
        error: message.into(),
    }
}

struct Lock(CVPixelBufferRef);
impl Drop for Lock {
    fn drop(&mut self) {
        unsafe {
            CVPixelBufferUnlockBaseAddress(self.0, kCVPixelBufferLock_ReadOnly);
        }
    }
}

fn pack_rows(
    input: &[u8],
    stride: usize,
    row_bytes: usize,
    bgra_to_rgb: bool,
) -> Result<Vec<u8>, NokhwaError> {
    let rows = input.len() / stride;
    let out_row = if bgra_to_rgb {
        row_bytes / BGRA_BYTES_PER_PIXEL * RGB_BYTES_PER_PIXEL
    } else {
        row_bytes
    };

    let mut out = Vec::new();
    out.try_reserve_exact(out_row * rows)
        .map_err(|_| error("Unable to allocate frame"))?;

    for line in input.chunks_exact(stride) {
        let pixels = &line[..row_bytes];
        if bgra_to_rgb {
            for bgra in pixels.chunks_exact(BGRA_BYTES_PER_PIXEL) {
                out.extend_from_slice(&[bgra[2], bgra[1], bgra[0]]);
            }
        } else {
            out.extend_from_slice(pixels);
        }
    }

    Ok(out)
}

#[allow(non_upper_case_globals)]
pub(crate) unsafe fn read(sample: CMSampleBufferRef) -> Result<DeliveredFrame, NokhwaError> {
    if sample.is_null() {
        return Err(error("AVFoundation delivered a null sample buffer"));
    }

    let buffer = CMSampleBufferGetImageBuffer(sample);
    if buffer.is_null() {
        return Err(error("AVFoundation delivered no pixel buffer; compressed sample output is not supported by this callback"));
    }

    let status = CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    if status != 0 {
        return Err(error(format!(
            "CVPixelBufferLockBaseAddress failed: {status}"
        )));
    }

    let _lock = Lock(buffer);

    let width = CVPixelBufferGetWidth(buffer);
    let height = CVPixelBufferGetHeight(buffer);
    let resolution = match (u32::try_from(width), u32::try_from(height)) {
        (Ok(w), Ok(h)) if w > 0 && h > 0 => Resolution::new(w, h),
        _ => return Err(error("Invalid pixel buffer dimensions")),
    };

    let fcc = CVPixelBufferGetPixelFormatType(buffer);
    let (bytes_per_pixel, format) = match fcc {
        kCVPixelFormatType_32BGRA => (BGRA_BYTES_PER_PIXEL, FrameFormat::RAWRGB),
        kCMPixelFormat_422YpCbCr8_yuvs if width % 2 == 0 => {
            (YUYV_BYTES_PER_PIXEL, FrameFormat::YUYV)
        }
        _ => {
            return Err(error(format!(
                "Unsupported delivered pixel format 0x{fcc:08x}; expected BGRA or yuvs"
            )))
        }
    };

    let row_bytes = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| error("Row size overflow"))?;
    let stride = CVPixelBufferGetBytesPerRow(buffer);
    let ptr = CVPixelBufferGetBaseAddress(buffer).cast::<u8>();
    if CVPixelBufferGetPlaneCount(buffer) != 0 || ptr.is_null() || stride < row_bytes {
        return Err(error("Invalid packed pixel buffer layout"));
    }

    let len = stride
        .checked_mul(height)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| error("Pixel buffer size overflow"))?;
    let input = slice::from_raw_parts(ptr, len);

    Ok(DeliveredFrame {
        data: pack_rows(input, stride, row_bytes, fcc == kCVPixelFormatType_32BGRA)?,
        format,
        resolution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgra_rows_become_rgb_without_padding() {
        // 2x2 BGRA, stride 12 (4 bytes of row padding marked 0xEE).
        let input = [
            1, 2, 3, 255, 4, 5, 6, 255, 0xEE, 0xEE, 0xEE, 0xEE, //
            7, 8, 9, 255, 10, 11, 12, 255, 0xEE, 0xEE, 0xEE, 0xEE,
        ];
        assert_eq!(
            pack_rows(&input, 12, 8, true).unwrap(),
            [3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10]
        );
    }

    #[test]
    fn yuyv_rows_are_copied_without_padding() {
        let input = [1, 2, 3, 4, 0xEE, 0xEE, 5, 6, 7, 8, 0xEE, 0xEE];
        assert_eq!(
            pack_rows(&input, 6, 4, false).unwrap(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
    }
}
