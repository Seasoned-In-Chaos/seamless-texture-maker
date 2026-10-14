//! Small value types shared across the engine.

use seams_core::delight::DelightParams;
use seams_core::pbr::{MapKind, MaterialParams};
use seams_core::seamless::SeamlessParams;
use serde::{Deserialize, Serialize};

/// The step of the workflow the user is working in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Delight,
    #[default]
    Seamless,
    Material,
}

impl Stage {
    pub const ALL: [Stage; 3] = [Stage::Delight, Stage::Seamless, Stage::Material];

    pub fn label(self) -> &'static str {
        match self {
            Stage::Delight => "Delight",
            Stage::Seamless => "Seamless",
            Stage::Material => "Material",
        }
    }
}

/// An image the document can show and export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Channel {
    #[default]
    BaseColor,
    Map(MapKind),
}

impl Channel {
    pub const ALL: [Channel; 6] = [
        Channel::BaseColor,
        Channel::Map(MapKind::Normal),
        Channel::Map(MapKind::Roughness),
        Channel::Map(MapKind::AmbientOcclusion),
        Channel::Map(MapKind::Displacement),
        Channel::Map(MapKind::Opacity),
    ];

    pub fn label(self) -> &'static str {
        match self {
            Channel::BaseColor => "Base color",
            Channel::Map(kind) => kind.label(),
        }
    }

    /// Suffix appended to the source file name on export.
    pub fn file_suffix(self) -> &'static str {
        match self {
            Channel::BaseColor => "_seamless",
            Channel::Map(kind) => kind.file_suffix(),
        }
    }
}

/// Every adjustable setting of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    pub delight: DelightParams,
    pub seamless: SeamlessParams,
    pub material: MaterialParams,
}

/// What a quick save of the base colour does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SaveMode {
    /// Write a new file beside the source, with a `_seamless` suffix.
    #[default]
    NewFile,
    /// Replace the source file.
    Overwrite,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_have_distinct_suffixes() {
        let mut suffixes: Vec<&str> = Channel::ALL.iter().map(|c| c.file_suffix()).collect();
        suffixes.sort_unstable();
        suffixes.dedup();

        assert_eq!(suffixes.len(), Channel::ALL.len());
    }

    #[test]
    fn params_survive_a_json_round_trip() {
        let mut params = Params::default();
        params.seamless.splat.seed = 7;
        params.material.ao.blur_sharp = -4.0;

        let json = serde_json::to_string(&params).unwrap();

        assert_eq!(serde_json::from_str::<Params>(&json).unwrap(), params);
    }

    #[test]
    fn unknown_and_missing_fields_fall_back_to_defaults() {
        let parsed: Params =
            serde_json::from_str(r#"{"seamless":{"method":"splat"},"x":1}"#).unwrap();

        assert_eq!(parsed.seamless.method, seams_core::seamless::Method::Splat);
        assert_eq!(parsed.delight, DelightParams::default());
    }
}
