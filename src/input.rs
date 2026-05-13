// Note: A decent amount of this file was inspired/adapted from the `minifb` crate

use std::collections::HashSet;
use num_enum::TryFromPrimitive;
use crate::window::Window;
// TODO keyrepeat?

const KEY_COUNT: usize = Key::Count as usize;

#[repr(usize)]
#[derive(Debug, Eq, PartialEq, TryFromPrimitive, Clone, Copy)]
pub enum Key {
    Key0 = 0,
    Key1 = 1,
    Key2 = 2,
    Key3 = 3,
    Key4 = 4,
    Key5 = 5,
    Key6 = 6,
    Key7 = 7,
    Key8 = 8,
    Key9 = 9,
    A = 10,
    B = 11,
    C = 12,
    D = 13,
    E = 14,
    F = 15,
    G = 16,
    H = 17,
    I = 18,
    J = 19,
    K = 20,
    L = 21,
    M = 22,
    N = 23,
    O = 24,
    P = 25,
    Q = 26,
    R = 27,
    S = 28,
    T = 29,
    U = 30,
    V = 31,
    W = 32,
    X = 33,
    Y = 34,
    Z = 35,
    F1 = 36,
    F2 = 37,
    F3 = 38,
    F4 = 39,
    F5 = 40,
    F6 = 41,
    F7 = 42,
    F8 = 43,
    F9 = 44,
    F10 = 45,
    F11 = 46,
    F12 = 47,
    F13 = 48,
    F14 = 49,
    F15 = 50,
    Down = 51,
    Left = 52,
    Right = 53,
    Up = 54,
    Apostrophe = 55,
    Backquote = 56,
    Backslash = 57,
    Comma = 58,
    Equal = 59,
    LeftBracket = 60,
    Minus = 61,
    Period = 62,
    RightBracket = 63,
    Semicolon = 64,
    Slash = 65,
    Backspace = 66,
    Delete = 67,
    End = 68,
    Enter = 69,
    Escape = 70,
    Home = 71,
    Insert = 72,
    Menu = 73,
    PageDown = 74,
    PageUp = 75,
    Pause = 76,
    Space = 77,
    Tab = 78,
    NumLock = 79,
    CapsLock = 80,
    ScrollLock = 81,
    LeftShift = 82,
    RightShift = 83,
    LeftCtrl = 84,
    RightCtrl = 85,
    NumPad0 = 86,
    NumPad1 = 87,
    NumPad2 = 88,
    NumPad3 = 89,
    NumPad4 = 90,
    NumPad5 = 91,
    NumPad6 = 92,
    NumPad7 = 93,
    NumPad8 = 94,
    NumPad9 = 95,
    NumPadDot = 96,
    NumPadSlash = 97,
    NumPadAsterisk = 98,
    NumPadMinus = 99,
    NumPadPlus = 100,
    NumPadEnter = 101,
    LeftAlt = 102,
    RightAlt = 103,
    LeftSuper = 104,
    RightSuper = 105,
    Unknown = 106,
    Count = 107,
}

pub enum MouseButton {
    Left,
    Middle,
    Right,
}

pub enum MouseMode {
    Pass, // mouse coordinates from outside the window (can be negative)
    Clamp, // clamp coordinates to window dimensions
    Discard, // discard if mouse is outside the window
}

pub struct InputState {
    keys_down: [bool; KEY_COUNT], // keys currently down
    keys_pressed: [bool; KEY_COUNT], // keys pressed since last update
    keys_released: [bool; KEY_COUNT], // keys released since last update

    mouse_pos: Option<(f32, f32)>,
    mouse_pos_unscaled: Option<(f32, f32)>,
    window_size: (f32, f32), // used for MouseMode calculations

    mouse_buttons_down: (bool, bool, bool), // left, middle, right 
    scroll_wheel: Option<(f32, f32)>, // (x, y)
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn update(&mut self, window: &Window) {
        // grabs all input info from backend, called at start of every frame
        // NOTE: this function's signature and contents should adapt
        // to whatever windowing/input crate is currently in use
        // pub(crate) because it should only be used inside the library

        // reset arrays
        self.keys_down.fill(false);
        self.keys_pressed.fill(false);
        self.keys_released.fill(false);

        // keyboard inputs
        // NOTE: this relies on our Key enum matching minifb's Key enum
        for key in window.get_keys() {
            self.keys_down[key as usize] = true;
        }
        for key in window.get_keys_pressed(minifb::KeyRepeat::No) {
            self.keys_pressed[key as usize] = true;
        }
        for key in window.get_keys_released() {
            self.keys_released[key as usize] = true;
        }

        // mouse/scroll inputs
        self.mouse_pos = window.get_mouse_pos(minifb::MouseMode::Pass);
        self.mouse_pos_unscaled = window.get_unscaled_mouse_pos(minifb::MouseMode::Pass);

        self.mouse_buttons_down = (
            window.get_mouse_down(minifb::MouseButton::Left),
            window.get_mouse_down(minifb::MouseButton::Middle),
            window.get_mouse_down(minifb::MouseButton::Right)
        );
        self.scroll_wheel = window.get_scroll_wheel();


    }

    // gets the keys that are currently held down
    pub fn get_keys(&self) -> Vec<Key> {
        self.keys_down
            .iter()
            .enumerate()
            .filter(|(_, down)| **down)
            .filter_map(|(index, _)| Key::try_from(index).ok())
            .collect()
    }

    // gets the keys that were pressed since last update
    pub fn get_keys_pressed(&self) -> Vec<Key> {
        self.keys_pressed
            .iter()
            .enumerate()
            .filter(|(_, down)| **down)
            .filter_map(|(index, _)| Key::try_from(index).ok())
            .collect()
    }

    // gets the keys that were released since last update
    pub fn get_keys_released(&self) -> Vec<Key> {
        self.keys_released
            .iter()
            .enumerate()
            .filter(|(_, down)| **down)
            .filter_map(|(index, _)| Key::try_from(index).ok())
            .collect()
    }

    // checks if a single key is held down
    pub fn is_key_down(&self, key: Key) -> bool {
        self.keys_down[key as usize]
    }

    // checks if a single key was pressed since last update
    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.keys_pressed[key as usize]
    }

    // checks if a single key has been released since last update
    pub fn is_key_released(&self, key: Key) -> bool {
        self.keys_released[key as usize]
    }

    // gets the current mouse position in the window (origin = top left)
    pub fn get_mouse_pos(&self, mode: MouseMode) -> Option<(f32, f32)> {
        let (x, y) = self.mouse_pos?;
        let (w, h) = self.window_size;

        match mode {
            MouseMode::Pass => Some((x, y)),
            MouseMode::Clamp => Some((x.clamp(0.0, w - 1.0), y.clamp(0.0, h - 1.0))),
            MouseMode::Discard => {
                if x < 0.0 || y < 0.0 || x >= w || y >= h {
                    None
                } else {
                    Some((x, y))
                }
            }
        }
    }

    // gets the current mouse position in the window (origin = top left)
    // ignores window scaling
    pub fn get_mouse_pos_unscaled(&self, mode: MouseMode) -> Option<(f32, f32)> {
        let (x, y) = self.mouse_pos_unscaled?;
        let (w, h) = self.window_size;

        match mode {
            MouseMode::Pass => Some((x, y)),
            MouseMode::Clamp => Some((x.clamp(0.0, w - 1.0), y.clamp(0.0, h - 1.0))),
            MouseMode::Discard => {
                if x < 0.0 || y < 0.0 || x >= w || y >= h {
                    None
                } else {
                    Some((x, y))
                }
            }
        }
    }

    // checks if a mouse button is down or not
    pub fn get_mouse_down(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Left => self.mouse_buttons_down.0,
            MouseButton::Middle => self.mouse_buttons_down.1,
            MouseButton::Right => self.mouse_buttons_down.2,
        }
    }

    // gets the current movement of the scroll wheel
    pub fn get_scroll_wheel(&self) -> Option<(f32, f32)> {
        self.scroll_wheel
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            keys_down: [false; KEY_COUNT],
            keys_pressed: [false; KEY_COUNT],
            keys_released: [false; KEY_COUNT],
            mouse_pos: None,
            mouse_pos_unscaled: None,
            window_size: (1.0, 1.0),
            mouse_buttons_down: (false, false, false),
            scroll_wheel: None,
        }
    }
}