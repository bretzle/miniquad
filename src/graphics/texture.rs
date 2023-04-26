use super::GlowTexture;
use crate::Context;
use glow::{
    HasContext, ALPHA, COLOR_ATTACHMENT0, DRAW_FRAMEBUFFER_BINDING, FRAMEBUFFER, RED, TEXTURE_2D,
    TEXTURE_MAG_FILTER, TEXTURE_MIN_FILTER, TEXTURE_SWIZZLE_A, TEXTURE_WRAP_S, TEXTURE_WRAP_T,
    UNPACK_ALIGNMENT,
};
use std::mem::transmute;

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct Texture {
    pub(crate) texture: Option<GlowTexture>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

impl Texture {
    pub fn empty() -> Self {
        Self {
            texture: None,
            width: 0,
            height: 0,
            format: TextureFormat::RGBA8,
        }
    }

    // pub fn gl_internal_id(&self) -> GLuint {
    //     self.texture
    // }

    // pub unsafe fn from_raw_id(texture: GLuint) -> Self {
    //     Self {
    //         texture,
    //         width: 0,
    //         height: 0,
    //         format: TextureFormat::RGBA8, // assumed for now
    //     }
    // }

    /// Delete GPU texture, leaving handle unmodified.
    ///
    /// More high-level code on top of miniquad probably is going to call this in Drop implementation of some
    /// more RAII buffer object.
    ///
    /// There is no protection against using deleted textures later. However its not an UB in OpenGl and thats why
    /// this function is not marked as unsafe
    pub fn delete(&mut self, gl: &glow::Context) {
        unsafe { gl.delete_texture(self.texture.take().unwrap()) }
    }
}

/// List of all the possible formats of input data when uploading to texture.
/// The list is built by intersection of texture formats supported by 3.3 core profile and webgl1.
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum TextureFormat {
    RGB8,
    RGBA8,
    Depth,
    Alpha,
    Rgba1555Rev,
}

/// Converts from TextureFormat to (internal_format, format, pixel_type)
impl From<TextureFormat> for (u32, u32, u32) {
    fn from(format: TextureFormat) -> Self {
        match format {
            TextureFormat::RGB8 => (glow::RGB, glow::RGB, glow::UNSIGNED_BYTE),
            TextureFormat::RGBA8 => (glow::RGBA, glow::RGBA, glow::UNSIGNED_BYTE),
            TextureFormat::Depth => (
                glow::DEPTH_COMPONENT,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_SHORT,
            ),
            #[cfg(target_arch = "wasm32")]
            TextureFormat::Alpha => (ALPHA, ALPHA, glow::UNSIGNED_BYTE),
            #[cfg(not(target_arch = "wasm32"))]
            TextureFormat::Alpha => (glow::R8, RED, glow::UNSIGNED_BYTE), // texture updates will swizzle Red -> Alpha to match WASM
            TextureFormat::Rgba1555Rev => {
                (glow::RGBA, glow::RGBA, glow::UNSIGNED_SHORT_1_5_5_5_REV)
            }
        }
    }
}
impl TextureFormat {
    /// Returns the size in bytes of texture with `dimensions`.
    pub fn size(self, width: u32, height: u32) -> u32 {
        let square = width * height;
        match self {
            TextureFormat::RGB8 => 3 * square,
            TextureFormat::RGBA8 => 4 * square,
            TextureFormat::Depth => 2 * square,
            TextureFormat::Alpha => 1 * square,
            TextureFormat::Rgba1555Rev => 2 * square,
        }
    }
}

impl Default for TextureParams {
    fn default() -> Self {
        TextureParams {
            format: TextureFormat::RGBA8,
            wrap: TextureWrap::Clamp,
            filter: FilterMode::Linear,
            width: 0,
            height: 0,
        }
    }
}

/// Sets the wrap parameter for texture.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TextureWrap {
    /// Samples at coord x + 1 map to coord x.
    Repeat = glow::REPEAT as isize,
    /// Samples at coord x + 1 map to coord 1 - x.
    Mirror = glow::MIRRORED_REPEAT as isize,
    /// Samples at coord x + 1 map to coord 1.
    Clamp = glow::CLAMP_TO_EDGE as isize,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum FilterMode {
    Linear = glow::LINEAR as isize,
    Nearest = glow::NEAREST as isize,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TextureAccess {
    /// Used as read-only from GPU
    Static,
    /// Can be written to from GPU
    RenderTarget,
}

#[derive(Debug, Copy, Clone)]
pub struct TextureParams {
    pub format: TextureFormat,
    pub wrap: TextureWrap,
    pub filter: FilterMode,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    /// Shorthand for `new(ctx, TextureAccess::RenderTarget, params)`
    pub fn new_render_texture(ctx: &mut Context, params: TextureParams) -> Texture {
        Self::new(ctx, TextureAccess::RenderTarget, None, params)
    }

    pub fn new(
        ctx: &mut Context,
        _access: TextureAccess,
        bytes: Option<&[u8]>,
        params: TextureParams,
    ) -> Self {
        if let Some(bytes_data) = bytes {
            assert_eq!(
                params.format.size(params.width, params.height) as usize,
                bytes_data.len()
            );
        }

        let (internal_format, format, pixel_type) = params.format.into();

        ctx.cache.store_texture_binding(0);

        unsafe {
            let texture = ctx.gl.create_texture().ok();

            ctx.cache.bind_texture(&ctx.gl, 0, texture);
            ctx.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1); // miniquad always uses row alignment of 1

            ctx.gl.tex_image_2d(
                TEXTURE_2D,
                0,
                internal_format as i32,
                params.width as i32,
                params.height as i32,
                0,
                format,
                pixel_type,
                bytes,
            );

            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, params.wrap as i32);
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, params.wrap as i32);
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, params.filter as i32);
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, params.filter as i32);

            if cfg!(not(target_arch = "wasm32")) {
                // if not WASM
                if params.format == TextureFormat::Alpha {
                    // if alpha miniquad texture, the value on non-WASM is stored in red channel
                    // swizzle red -> alpha
                    ctx.gl
                        .tex_parameter_i32(TEXTURE_2D, TEXTURE_SWIZZLE_A, RED as _);
                } else {
                    // keep alpha -> alpha
                    ctx.gl
                        .tex_parameter_i32(TEXTURE_2D, TEXTURE_SWIZZLE_A, ALPHA as _);
                }
            }
            ctx.cache.restore_texture_binding(&ctx.gl, 0);

            Self {
                texture,
                width: params.width,
                height: params.height,
                format: params.format,
            }
        }
    }

    /// Upload texture to GPU with given TextureParams
    pub fn from_data_and_format(ctx: &mut Context, bytes: &[u8], params: TextureParams) -> Texture {
        Self::new(ctx, TextureAccess::Static, Some(bytes), params)
    }

    /// Upload RGBA8 texture to GPU
    pub fn from_rgba8(ctx: &mut Context, width: u16, height: u16, bytes: &[u8]) -> Texture {
        assert_eq!(width as usize * height as usize * 4, bytes.len());

        Self::from_data_and_format(
            ctx,
            bytes,
            TextureParams {
                width: width as _,
                height: height as _,
                format: TextureFormat::RGBA8,
                wrap: TextureWrap::Clamp,
                filter: FilterMode::Linear,
            },
        )
    }

    /// Set the min and mag filter to `filter`
    pub fn set_filter(&self, ctx: &mut Context, filter: FilterMode) {
        ctx.cache.store_texture_binding(0);
        ctx.cache.bind_texture(&ctx.gl, 0, self.texture);
        unsafe {
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, filter as i32);
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, filter as i32);
        }
        ctx.cache.restore_texture_binding(&ctx.gl, 0);
    }

    /// Set the min and mag filter separately
    pub fn set_filter_min_mag(
        &self,
        ctx: &mut Context,
        min_filter: FilterMode,
        mag_filter: FilterMode,
    ) {
        ctx.cache.store_texture_binding(0);
        ctx.cache.bind_texture(&ctx.gl, 0, self.texture);
        unsafe {
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, min_filter as i32);
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, mag_filter as i32);
        }
        ctx.cache.restore_texture_binding(&ctx.gl, 0);
    }

    /// Set x and y wrap to `wrap`
    pub fn set_wrap(&self, ctx: &mut Context, wrap: TextureWrap) {
        ctx.cache.store_texture_binding(0);
        ctx.cache.bind_texture(&ctx.gl, 0, self.texture);
        unsafe {
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, wrap as i32);
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, wrap as i32);
        }
        ctx.cache.restore_texture_binding(&ctx.gl, 0);
    }

    /// Set x and y wrap separately
    pub fn set_wrap_xy(&self, ctx: &mut Context, x_wrap: TextureWrap, y_wrap: TextureWrap) {
        ctx.cache.store_texture_binding(0);
        ctx.cache.bind_texture(&ctx.gl, 0, self.texture);
        unsafe {
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, x_wrap as i32);
            ctx.gl
                .tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, y_wrap as i32);
        }
        ctx.cache.restore_texture_binding(&ctx.gl, 0);
    }

    pub fn resize(&mut self, ctx: &mut Context, width: u32, height: u32, bytes: Option<&[u8]>) {
        ctx.cache.store_texture_binding(0);
        ctx.cache.bind_texture(&ctx.gl, 0, self.texture);

        let (internal_format, format, pixel_type) = self.format.into();

        self.width = width;
        self.height = height;

        unsafe {
            ctx.gl.pixel_store_i32(UNPACK_ALIGNMENT, 1); // miniquad always uses row alignment of 1

            ctx.gl.tex_image_2d(
                TEXTURE_2D,
                0,
                internal_format as i32,
                self.width as i32,
                self.height as i32,
                0,
                format,
                pixel_type,
                bytes,
            );
        }

        ctx.cache.restore_texture_binding(&ctx.gl, 0);
    }

    /// Update whole texture content
    /// bytes should be width * height * 4 size - non rgba8 textures are not supported yet anyway
    pub fn update(&self, ctx: &mut Context, bytes: &[u8]) {
        assert_eq!(self.size(self.width, self.height), bytes.len());

        self.update_texture_part(
            ctx,
            0 as _,
            0 as _,
            self.width as _,
            self.height as _,
            bytes,
        )
    }

    pub fn update_texture_part(
        &self,
        ctx: &mut Context,
        x_offset: i32,
        y_offset: i32,
        width: i32,
        height: i32,
        bytes: &[u8],
    ) {
        assert_eq!(self.size(width as _, height as _), bytes.len());
        assert!(x_offset + width <= self.width as _);
        assert!(y_offset + height <= self.height as _);

        ctx.cache.store_texture_binding(0);
        ctx.cache.bind_texture(&ctx.gl, 0, self.texture);

        let (_, format, pixel_type) = self.format.into();

        unsafe {
            ctx.gl.pixel_store_i32(UNPACK_ALIGNMENT, 1); // miniquad always uses row alignment of 1

            ctx.gl.tex_sub_image_2d(
                TEXTURE_2D,
                0,
                x_offset as _,
                y_offset as _,
                width as _,
                height as _,
                format,
                pixel_type,
                glow::PixelUnpackData::Slice(bytes),
            );
        }

        ctx.cache.restore_texture_binding(&ctx.gl, 0);
    }

    /// Read texture data into CPU memory
    pub fn read_pixels(&self, ctx: &mut Context, bytes: &mut [u8]) {
        if self.format == TextureFormat::Alpha {
            unimplemented!("read_pixels is not implement for Alpha textures");
        }
        let (_, format, pixel_type) = self.format.into();

        unsafe {
            let binded_fbo = ctx.gl.get_parameter_i32(DRAW_FRAMEBUFFER_BINDING);
            let fbo = ctx.gl.create_framebuffer().ok();
            ctx.gl.bind_framebuffer(FRAMEBUFFER, fbo);
            ctx.gl.framebuffer_texture_2d(
                FRAMEBUFFER,
                COLOR_ATTACHMENT0,
                TEXTURE_2D,
                self.texture,
                0,
            );

            ctx.gl.read_pixels(
                0,
                0,
                self.width as _,
                self.height as _,
                format,
                pixel_type,
                glow::PixelPackData::Slice(bytes),
            );

            ctx.gl.bind_framebuffer(FRAMEBUFFER, transmute(binded_fbo));
            ctx.gl.delete_framebuffer(fbo.unwrap());
        }
    }

    #[inline]
    fn size(&self, width: u32, height: u32) -> usize {
        self.format.size(width, height) as usize
    }
}
