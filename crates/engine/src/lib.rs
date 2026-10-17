//! The SEAMS application engine.
//!
//! Everything the app does lives here, with no UI toolkit in sight: the open
//! document and its undo history, the render pipeline and its caches, the
//! background workers and the persisted settings. A front end only has to
//! forward user input to a [`Session`] and draw what it exposes.

mod display;
mod document;
mod error;
mod model;
mod pipeline;
mod session;
mod settings;
mod worker;

pub use display::DisplayImage;
pub use document::Document;
pub use error::{EngineError, Result};
pub use model::{Channel, Params, SaveMode, Stage};
pub use session::{Activity, DISPLAY_MAX_SIDE, Event, Notify, Session, View};
pub use settings::{RetouchSettings, Settings, ViewSettings, WindowSettings, app_data_dir};
