use minifb::WindowOptions;

pub struct Window {
    // wraps minifb's Window to add custom functionality
    pub inner: minifb::Window,
}

impl Window {
    pub fn new(name: &str, width: usize, height: usize, options: WindowOptions) -> Result<Self, minifb::Error> {
        let inner = minifb::Window::new(name, width, height, options)?;
        Ok(Self {
            inner,
        })
    }

    pub fn set_mouse_pos(&mut self, x: f32, y: f32) {
        // set cursor position relative to the window using raw window handle
        #[cfg(target_os = "windows")] // only windows for now
        use winapi::um::winuser::{SetCursorPos, ClientToScreen};
        use winapi::shared::windef::{HWND, POINT};
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
        unsafe {
            // get the raw window handle
            if let Ok(window_handle) = self.inner.window_handle()
                && let RawWindowHandle::Win32(handle) = window_handle.as_raw() {
                let hwnd = handle.hwnd.get() as HWND;
                
                let mut point = POINT {
                    x: x as i32,
                    y: y as i32,
                };

                // convert client coordinates to screen coordinates
                ClientToScreen(hwnd, &mut point);
                SetCursorPos(point.x, point.y);
            }
        }
    }
}

// Deref + DerefMut -> minifb window can be accessed directly
impl std::ops::Deref for Window {
    type Target = minifb::Window;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for Window {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}