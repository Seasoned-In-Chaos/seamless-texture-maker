//! A small file logger.
//!
//! Writes to `logs/seams.log` in the app data folder. If that cannot be
//! opened the app still runs; it just has nowhere to write diagnostics.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use log::{Level, LevelFilter, Log, Metadata, Record};

const LOG_FILE: &str = "seams.log";
/// The log is restarted once it grows past this, so it cannot fill the disk.
const MAX_LOG_BYTES: u64 = 1024 * 1024;

struct FileLogger {
    file: Mutex<File>,
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let seconds =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let mut file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        // A failed log write has nowhere better to be reported.
        let _ = writeln!(
            file,
            "{seconds} {:5} [{}] {}",
            record.level(),
            record.target(),
            record.args()
        );
    }

    fn flush(&self) {
        let _ = self.file.lock().unwrap_or_else(PoisonError::into_inner).flush();
    }
}

fn open_log(folder: &Path) -> std::io::Result<File> {
    std::fs::create_dir_all(folder)?;
    let path = folder.join(LOG_FILE);
    let too_big = std::fs::metadata(&path).map(|m| m.len() > MAX_LOG_BYTES).unwrap_or(false);
    OpenOptions::new().create(true).append(!too_big).write(true).truncate(too_big).open(path)
}

fn log_folder() -> Option<PathBuf> {
    seams_engine::app_data_dir().map(|dir| dir.join("logs"))
}

/// Starts logging to the app data folder and records panics there.
pub fn init() {
    let Some(file) = log_folder().and_then(|folder| open_log(&folder).ok()) else { return };
    let logger = FileLogger { file: Mutex::new(file) };
    if log::set_boxed_logger(Box::new(logger)).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
    std::panic::set_hook(Box::new(|info| log::error!("panic: {info}")));
}
