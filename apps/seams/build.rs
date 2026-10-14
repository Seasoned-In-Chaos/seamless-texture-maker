//! Embeds the icon and version information in `seams.exe`.

use std::path::PathBuf;
use std::process::Command;

const ICON: &str = "../../resources/icon.ico";

fn main() {
    println!("cargo:rerun-if-changed={ICON}");
    // Absolute, because the resource compiler runs from another directory and
    // would not find a path relative to this crate.
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from);
    let Some(icon) = manifest_dir.map(|dir| dir.join(ICON)).filter(|icon| icon.exists()) else {
        return;
    };
    if std::env::var_os("CARGO_CFG_WINDOWS").is_none() {
        return;
    }
    let mut resource = winresource::WindowsResource::new();
    resource.set_icon(&icon.to_string_lossy());
    resource.set("ProductName", "SEAMS");
    resource.set("FileDescription", "SEAMS - Seamless Texture Studio");
    resource.set("LegalCopyright", "Copyright (c) Shubham Panchasara. MIT license.");
    let result = if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu") {
        compile_with_windres(&resource)
    } else {
        resource.compile().map_err(|error| error.to_string())
    };
    // A missing resource compiler only costs the exe its icon: warn, don't fail.
    if let Err(error) = result {
        println!("cargo:warning=could not embed the application icon: {error}");
    }
}

/// `winresource` passes the crate directory to `windres` as an include path,
/// and `windres` hands it to its preprocessor unquoted, so a project folder
/// with a space in its name fails. The script includes nothing, so it is
/// compiled here without one, by relative name from inside `OUT_DIR`.
fn compile_with_windres(resource: &winresource::WindowsResource) -> Result<(), String> {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?);
    resource.write_resource_file(out_dir.join("resource.rc")).map_err(|error| error.to_string())?;
    let windres = std::env::var("WINDRES").unwrap_or_else(|_| "windres".to_owned());
    let status = Command::new(&windres)
        .current_dir(&out_dir)
        .args(["resource.rc", "resource.o"])
        .status()
        .map_err(|error| format!("could not run {windres}: {error}"))?;
    if !status.success() {
        return Err(format!("{windres} failed ({status})"));
    }
    // An object file on the link line is always linked in, resources included.
    println!("cargo:rustc-link-arg-bins={}", out_dir.join("resource.o").display());
    Ok(())
}
