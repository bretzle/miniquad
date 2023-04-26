#![allow(non_snake_case)]

pub const GL_TIME_ELAPSED: u32 = 35007;

pub unsafe fn glGetQueryObjectui64v(_id: u32, _pname: u32, _params: *mut u64) {
    unimplemented!();
}

pub unsafe fn glGetQueryObjectiv(_id: u32, _pname: u32, _params: *mut i32) {
    unimplemented!();
}
