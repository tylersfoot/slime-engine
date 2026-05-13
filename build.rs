// this build script gets run before the rest of the code is compiled
// it copies the `res/` into the output directory 
fn main() -> anyhow::Result<()> {
    // this tells Cargo to rerun this script if something in /res/ changes
    println!("cargo:rerun-if-changed=res");

    let out_dir = std::env::var("OUT_DIR")?;
    let mut copy_options = fs_extra::dir::CopyOptions::new();
    copy_options.overwrite = true;
    let paths_to_copy = vec!["res/"];

    // wipe folder to fully sync files
    let dest_path = std::path::Path::new(&out_dir).join("res");
    if dest_path.exists() {
        std::fs::remove_dir_all(&dest_path)?;
    }
    fs_extra::copy_items(&paths_to_copy, out_dir, &copy_options)?;

    Ok(())
}
