fn main() {
    #[cfg(target_os = "windows")]
    {
        // Link MKL components in the correct order
        println!("cargo:rustc-link-lib=static=mkl_blas95_lp64");
        println!("cargo:rustc-link-lib=static=mkl_lapack95_lp64");
        println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=static=mkl_sequential");
        println!("cargo:rustc-link-lib=static=mkl_core");
        
        // Windows-specific linker flags
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:LIBCMT");
        
        // Use the MKL library search path from an environment variable
        if let Ok(mkl_lib_path) = std::env::var("MKL_LIB_PATH") {
            println!("cargo:rustc-link-search={}", mkl_lib_path);
        } else {
            eprintln!("Error: MKL_LIB_PATH environment variable not set.");
            std::process::exit(1);
        }
    }
}
