//! The open texture and its undo history.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use seams_core::Image;
use seams_io::ImageMeta;

/// Undo steps kept. Each one is a full-resolution image, so the history is
/// capped by count and by memory; the oldest step is dropped first.
const MAX_UNDO_STEPS: usize = 20;
/// Memory the undo history may hold. The most recent step is always kept,
/// even when it alone is larger than this.
const MAX_UNDO_BYTES: usize = 1536 * 1024 * 1024;

/// The texture being edited.
///
/// `base` is the current base colour: the loaded image, or the result of the
/// last committed edit. Rendering never changes it; only [`Document::commit`],
/// undo and redo do.
#[derive(Debug, Clone)]
pub struct Document {
    path: PathBuf,
    meta: ImageMeta,
    base: Arc<Image>,
    undo: Vec<Arc<Image>>,
    redo: Vec<Arc<Image>>,
}

impl Document {
    pub fn new(path: PathBuf, meta: ImageMeta, image: Image) -> Self {
        Self { path, meta, base: Arc::new(image), undo: Vec::new(), redo: Vec::new() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file name without its folder, for titles and status text.
    pub fn file_name(&self) -> String {
        self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
    }

    pub fn meta(&self) -> &ImageMeta {
        &self.meta
    }

    pub fn base(&self) -> &Arc<Image> {
        &self.base
    }

    /// Replaces the base colour, remembering the previous one for undo.
    pub fn commit(&mut self, image: Arc<Image>) {
        self.undo.push(std::mem::replace(&mut self.base, image));
        self.redo.clear();
        self.trim_history(MAX_UNDO_STEPS, MAX_UNDO_BYTES);
    }

    /// Drops the oldest undo steps until both limits hold.
    fn trim_history(&mut self, max_steps: usize, max_bytes: usize) {
        let held = |steps: &[Arc<Image>]| steps.iter().map(|step| step.byte_len()).sum::<usize>();
        while self.undo.len() > max_steps || (self.undo.len() > 1 && held(&self.undo) > max_bytes) {
            self.undo.remove(0);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Steps back; returns false when there is nothing to undo.
    pub fn undo(&mut self) -> bool {
        let Some(previous) = self.undo.pop() else { return false };
        self.redo.push(std::mem::replace(&mut self.base, previous));
        true
    }

    /// Steps forward; returns false when there is nothing to redo.
    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo.pop() else { return false };
        self.undo.push(std::mem::replace(&mut self.base, next));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seams_io::{BitDepth, FileFormat};

    fn image(value: f32) -> Image {
        Image::filled(64, 64, 3, value).unwrap()
    }

    fn document() -> Document {
        let meta = ImageMeta {
            width: 64,
            height: 64,
            channels: 3,
            bit_depth: BitDepth::Eight,
            format: FileFormat::Png,
            dpi: None,
            file_size: 0,
        };
        Document::new(PathBuf::from("folder/rock.png"), meta, image(0.1))
    }

    fn base_value(doc: &Document) -> f32 {
        doc.base().data()[0]
    }

    #[test]
    fn new_document_has_no_history() {
        let doc = document();

        assert!(!doc.can_undo() && !doc.can_redo());
        assert_eq!(doc.file_name(), "rock.png");
    }

    #[test]
    fn undo_and_redo_walk_the_history() {
        let mut doc = document();
        doc.commit(Arc::new(image(0.2)));
        doc.commit(Arc::new(image(0.3)));

        assert!(doc.undo());
        assert_eq!(base_value(&doc), 0.2);
        assert!(doc.undo());
        assert_eq!(base_value(&doc), 0.1);
        assert!(!doc.undo());
        assert!(doc.redo());
        assert_eq!(base_value(&doc), 0.2);
    }

    #[test]
    fn committing_clears_redo() {
        let mut doc = document();
        doc.commit(Arc::new(image(0.2)));
        doc.undo();

        doc.commit(Arc::new(image(0.9)));

        assert!(!doc.can_redo());
        assert!(!doc.redo());
    }

    #[test]
    fn history_is_capped() {
        let mut doc = document();
        for step in 0..MAX_UNDO_STEPS + 5 {
            doc.commit(Arc::new(image(step as f32 / 100.0)));
        }

        let undone = std::iter::from_fn(|| doc.undo().then_some(())).count();

        assert_eq!(undone, MAX_UNDO_STEPS);
    }

    #[test]
    fn history_respects_the_memory_budget_but_keeps_the_latest_step() {
        let step_bytes = image(0.0).byte_len();
        let mut doc = document();
        for step in 0..6 {
            doc.commit(Arc::new(image(step as f32 / 10.0)));
        }

        doc.trim_history(MAX_UNDO_STEPS, step_bytes * 2 + 1);
        assert_eq!(doc.undo.len(), 2);

        doc.trim_history(MAX_UNDO_STEPS, 1);
        assert_eq!(doc.undo.len(), 1, "one step always survives");
    }
}
